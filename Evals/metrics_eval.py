import json
from typing import Any, Dict, List, Tuple, Optional
import re
import unicodedata
import jiwer
import spacy
from sklearn.metrics import precision_recall_fscore_support

# --------- Carga de JSON -> texto (soporta 'turns') ---------
def _join_turns(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items:
        if not isinstance(it, dict): 
            continue
        txt = (it.get("text") or it.get("content") or "").strip()
        if not txt:
            continue
        spk = None
        for k in ("speaker","role","author","name"):
            v = it.get(k)
            if isinstance(v, str) and v.strip():
                spk = v.strip(); break
        lines.append(f"{spk}: {txt}" if spk else txt)
    return "\n".join(lines).strip()

def _extract_text_recursive(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("turns","messages","utterances","dialogue","conversation"):
            if key in obj and isinstance(obj[key], list):
                j = _join_turns(obj[key])
                if j: return j
        for f in ("text","content","transcript","transcription","message"):
            v = obj.get(f)
            if isinstance(v, str) and v.strip(): 
                return v.strip()
        for v in obj.values():
            if isinstance(v, (dict,list)):
                found = _extract_text_recursive(v)
                if found: return found
    elif isinstance(obj, list):
        j = _join_turns([x for x in obj if isinstance(x, dict)])
        if j: return j
        for v in obj:
            found = _extract_text_recursive(v)
            if found: return found
    return ""

def load_dialogue_as_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = _extract_text_recursive(data).strip()
    if not text and isinstance(data, dict):
        # fallback: concatenar strings largas de primer nivel
        text = " ".join([v.strip() for v in data.values() if isinstance(v, str) and len(v.strip())>10])
    return text

# --------- Normalización básica ---------
def norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

# --------- Vocabulario clínico (para KER / MC-WER) ---------
def load_vocab(path: Optional[str]) -> List[str]:
    if not path:
        # Baseline mínimo; reemplázalo por tu lista
        return [
            "antihipertensivo","antihistamínico","antihistaminico","betabloqueante",
            "ieca","ara ii","ara-ii","calcioantagonista","estatina",
            "paracetamol","ibuprofeno","amoxicilina","insulina","metformina",
            "mercurio","omeprazol"
        ]
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if not term: continue
            # permite CSV: coge primera columna
            term = term.split(",")[0].strip()
            terms.append(term)
    return terms

# Extraer términos del texto según vocabulario (con coincidencia por palabra)
def extract_terms(text: str, vocab: List[str]) -> List[str]:
    t = norm(text)
    found = []
    for term in vocab:
        tn = norm(term)
        # mismo criterio que MC-WER: límites de palabra
        pattern = r"\b" + re.escape(tn) + r"\b"
        if re.search(pattern, t):
            found.append(tn)
    return found

# --------- Métricas ---------
def compute_wer(ref: str, hyp: str) -> float:
    return jiwer.wer(ref, hyp)

def compute_cer(ref: str, hyp: str) -> float:
    return jiwer.cer(ref, hyp)

# KER = 1 - F1 sobre keywords (definido explícitamente así)
def compute_ker(ref: str, hyp: str, vocab: List[str]) -> Tuple[float, float, float, float]:
    ref_terms = set(extract_terms(ref, vocab))
    hyp_terms = set(extract_terms(hyp, vocab))
    all_terms = sorted(list(ref_terms.union(hyp_terms)))
    if not all_terms:
        # sin términos -> no definimos KER; devolvemos N/A vía -1
        return -1.0, 0.0, 0.0, 0.0
    y_true = [1 if t in ref_terms else 0 for t in all_terms]
    y_pred = [1 if t in hyp_terms else 0 for t in all_terms]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    ker = 1.0 - f1
    return ker, p, r, f1

# MC-WER: WER sobre la SECUENCIA de términos médicos (ordenados por aparición)
def compute_mc_wer(ref: str, hyp: str, vocab: List[str]) -> float:
    """
    MC-WER: WER solo sobre la SECUENCIA de términos clínicos,
    reutilizando el mismo extractor que KER para evitar desajustes.
    """
    # 1) términos detectados (mismo matcher que KER)
    ref_terms = extract_terms(ref, vocab)
    hyp_terms = extract_terms(hyp, vocab)

    # 2) reordenar por aparición real en el texto (por posiciones)
    def ordered_seq(text: str, terms: List[str]) -> List[str]:
        tnorm = norm(text)
        positions = []
        for term in set(terms):  # set para no repetir el mismo término si aparece varias veces en vocab
            pat = re.compile(r"\b" + re.escape(term) + r"\b")
            for m in pat.finditer(tnorm):
                positions.append((m.start(), term))
        positions.sort(key=lambda x: x[0])
        return [tok for _, tok in positions]

    ref_seq = ordered_seq(ref, ref_terms)
    hyp_seq = ordered_seq(hyp, hyp_terms)

    print("DEBUG MC-WER seqs →")
    print("  REF_seq:", ref_seq)
    print("  HYP_seq:", hyp_seq)


    if not ref_seq and not hyp_seq:
        return -1.0  # N/A (no hay términos clínicos en ninguno)
    return jiwer.wer(" ".join(ref_seq), " ".join(hyp_seq))


# NER-based F1 (si quieres además comparar con NER)
def ner_based_f1(ref: str, hyp: str, nlp) -> float:
    def ents(s: str) -> set:
        if "ner" not in nlp.pipe_names: return set()
        return set([norm(ent.text) for ent in nlp(s).ents if ent.text.strip()])
    ref_e = ents(ref); hyp_e = ents(hyp)
    all_e = sorted(list(ref_e.union(hyp_e)))
    if not all_e: return -1.0  # N/A
    y_true = [1 if e in ref_e else 0 for e in all_e]
    y_pred = [1 if e in hyp_e else 0 for e in all_e]
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return f1

# --------- Main ---------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Métricas clínicas para transcripciones (ES)")
    ap.add_argument("original")
    ap.add_argument("transcribed")
    ap.add_argument("--vocab", help="ruta a vocabulario clínico (txt/CSV, una entrada por línea)")
    ap.add_argument("--show", action="store_true", help="muestra preview")
    args = ap.parse_args()

    ref = load_dialogue_as_text(args.original)
    hyp = load_dialogue_as_text(args.transcribed)

    if args.show:
        print("—— Preview ——")
        print(f"len(ref)={len(ref)} | len(hyp)={len(hyp)}")
        print("REF[:200]:", ref[:200].replace("\n"," ⏎ "))
        print("HYP[:200]:", hyp[:200].replace("\n"," ⏎ "))
        print()

    vocab = load_vocab(args.vocab)

    ref_terms = extract_terms(ref, vocab)
    hyp_terms = extract_terms(hyp, vocab)
    print("DEBUG vocab hits →")
    print("  REF:", sorted(set(ref_terms)))
    print("  HYP:", sorted(set(hyp_terms)))
    

    print("=============================================")
    print("🏥 MÉTRICAS CLÍNICAS")
    print("=============================================")
    print(f"WER: {compute_wer(ref, hyp):.3f}")
    print(f"CER: {compute_cer(ref, hyp):.3f}")

    mcwer = compute_mc_wer(ref, hyp, vocab)
    if mcwer < 0:
        print("MC-WER: N/A (sin términos médicos)")
    else:
        print(f"MC-WER (WER solo en términos clínicos): {mcwer:.3f}")

    ker, p, r, f1 = compute_ker(ref, hyp, vocab)
    if ker < 0:
        print("KER (1−F1 keywords): N/A (sin términos en vocabulario)")
    else:
        print(f"KER (1−F1 keywords): {ker:.3f}  |  Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    # NER-F1 (opcional, informativo)
    try:
        nlp = spacy.load("es_core_news_md")
        ner_f1 = ner_based_f1(ref, hyp, nlp)
        if ner_f1 < 0:
            print("NER-based F1: N/A (sin entidades detectadas)")
        else:
            print(f"NER-based F1 (spaCy genérico): {ner_f1:.3f}")
    except Exception:
        print("NER-based F1: N/A (spaCy 'es_core_news_md' no disponible)")
