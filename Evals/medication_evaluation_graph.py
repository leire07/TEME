"""
LangGraph implementation for medication evaluation with multi-agent architecture.

This graph implements a two-layer evaluation system:
1. First layer: Three specialized agents analyze text independently
2. Second layer: Consensus agent combines decisions and provides final classification

Updated for LangChain 0.3.27, LangGraph 0.6.7 and related dependencies
Enhanced with detailed error explanations for GRAVE classifications
"""

from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# State definition
class EvaluationState(TypedDict):
    """State for the medication evaluation graph"""
    original_text: str
    transcribed_text: str
    medication_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    dosage_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    consistency_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    final_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    explanations: List[str]
    consensus_explanation: str
    # New fields for detailed error explanations
    medication_explanation: str
    dosage_explanation: str
    consistency_explanation: str
    error_details: List[str]


# Initialize LLM (lazy initialization to avoid import-time API key requirement)
def get_llm():
    """Get the LLM instance with API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=api_key
    )

def parse_classification(text: str) -> str:
    """
    Devuelve NINGUNA, LEVE o GRAVE si aparece en el texto (ignorando adornos).
    Si no aparece, retorna NINGUNA como fallback conservador.
    """
    m = re.search(r'\b(NINGUNA|LEVE|GRAVE)\b', text.upper())
    return m.group(1) if m else "NINGUNA"

def extract_explanation(response_text: str) -> str:
    """
    Extrae la explicación del error de la respuesta del LLM.
    Busca texto después de la clasificación.
    """
    lines = response_text.strip().split('\n')
    explanation_lines = []
    
    for line in lines:
        # Skip empty lines and classification-only lines
        line = line.strip()
        if line and not re.match(r'^\b(NINGUNA|LEVE|GRAVE)\b$', line.upper()):
            explanation_lines.append(line)
    
    return ' '.join(explanation_lines).strip()


def medication_agent(state: EvaluationState) -> EvaluationState:
    """MedicationAgent: Evaluates medication name fidelity"""

    llm = get_llm()

    prompt = """Eres un experto en medicina clínica y en terminología farmacológica.
Tu tarea es comparar el texto original con la transcripción y evaluar la fidelidad de los nombres de medicamentos.

Instrucciones:
• Considera que conoces los nombres comerciales y genéricos de los fármacos.
• Marca error solo si el medicamento cambia de identidad (por ejemplo, un fármaco distinto o de otra clase terapéutica).
• No marques como error diferencias de formato, abreviaturas o variantes de escritura si el significado clínico es el mismo.
• Haz recuento de los medicamentos mencionados en ambos textos y compáralos.
• Si el original menciona una CLASE farmacológica (p. ej., "antihipertensivo")
  y la transcripción menciona un término que NO es un fármaco (p. ej., "mercurio", "sodio"),
  clasifica como GRAVE.

Clasifica el resultado en una única categoría:
• NINGUNA → el medicamento es el mismo.
• LEVE → variación poco clara de escritura, pero se reconoce como el mismo medicamento.
• GRAVE → el medicamento transcrito corresponde a otro diferente, aparecen medicamentos no mencionados en el original.

IMPORTANTE: Si la clasificación es GRAVE o LEVE, proporciona una explicación detallada del error encontrado.

TEXTO ORIGINAL:
{original_text}

TEXTO TRANSCRITO:
{transcribed_text}

Formato de respuesta:
1. Primera línea: SOLO la categoría (NINGUNA, LEVE o GRAVE)
2. Si es GRAVE o LEVE: Líneas adicionales con explicación detallada del error específico encontrado."""

    response = llm.invoke([
        HumanMessage(content=prompt.format(
            original_text=state["original_text"],
            transcribed_text=state["transcribed_text"]
        ))
    ])

    response_text = response.content.strip()
    classification = parse_classification(response_text)
    explanation = extract_explanation(response_text) if classification in ["GRAVE", "LEVE"] else ""

    # Create new state with medication classification and explanation
    new_state = state.copy()
    new_state["medication_classification"] = classification
    new_state["medication_explanation"] = explanation
    return new_state


def dosage_agent(state: EvaluationState) -> EvaluationState:
    """DosageAgent: Evaluates dosage accuracy"""

    llm = get_llm()

    prompt = """Eres un experto en farmacología clínica y en posología.
Tu tarea es comparar el texto original con la transcripción y comprobar si la dosis está bien transcrita.

Instrucciones:
• Marca error solo si cambia la cantidad, la unidad o la frecuencia de la dosis.
• No marques como error diferencias de estilo o de formato si el significado es el mismo (ejemplo: "200 mg/día" y "200 miligramos al día").
• Presta especial atención a diferencias numéricas que puedan ser críticas para la seguridad del paciente.

Clasifica el resultado en una única categoría:
• NINGUNA → la dosis tiene el mismo significado.
• LEVE → hay una diferencia menor que puede generar ligera confusión, pero no cambia la dosis.
• GRAVE → la dosis, la unidad o la frecuencia han cambiado de forma significativa.

IMPORTANTE: Si la clasificación es GRAVE o LEVE, proporciona una explicación detallada del error encontrado.

TEXTO ORIGINAL:
{original_text}

TEXTO TRANSCRITO:
{transcribed_text}

Formato de respuesta:
1. Primera línea: SOLO la categoría (NINGUNA, LEVE o GRAVE)
2. Si es GRAVE o LEVE: Líneas adicionales con explicación detallada del error específico encontrado."""

    response = llm.invoke([
        HumanMessage(content=prompt.format(
            original_text=state["original_text"],
            transcribed_text=state["transcribed_text"]
        ))
    ])

    response_text = response.content.strip()
    classification = parse_classification(response_text)
    explanation = extract_explanation(response_text) if classification in ["GRAVE", "LEVE"] else ""

    # Create new state with dosage classification and explanation
    new_state = state.copy()
    new_state["dosage_classification"] = classification
    new_state["dosage_explanation"] = explanation
    return new_state


def consistency_agent(state: EvaluationState) -> EvaluationState:
    """ConsistencyAgent: Evaluates overall coherence"""

    llm = get_llm()

    prompt = """Eres un experto en redacción médica y en coherencia clínica.
Tu tarea es comparar el texto original con la transcripción y verificar si se mantiene la coherencia de la información (síntomas, diagnósticos, alergias, instrucciones).

Instrucciones:
• Marca error solo si cambia el sentido clínico.
• Ignora diferencias de estilo, pequeñas omisiones o reformulaciones que no alteran el significado.
• Presta especial atención a cambios que puedan afectar la seguridad del paciente o el diagnóstico.

Clasifica el resultado en una única categoría:
• NINGUNA → no hay cambios de significado clínico.
• LEVE → se omite o cambia un detalle secundario, sin afectar al sentido clínico principal.
• GRAVE → cambia el significado de forma importante (ejemplo: de "no tiene alergias" a "tiene alergias").

IMPORTANTE: Si la clasificación es GRAVE o LEVE, proporciona una explicación detallada del error encontrado.

TEXTO ORIGINAL:
{original_text}

TEXTO TRANSCRITO:
{transcribed_text}

Formato de respuesta:
1. Primera línea: SOLO la categoría (NINGUNA, LEVE o GRAVE)
2. Si es GRAVE o LEVE: Líneas adicionales con explicación detallada del error específico encontrado."""

    response = llm.invoke([
        HumanMessage(content=prompt.format(
            original_text=state["original_text"],
            transcribed_text=state["transcribed_text"]
        ))
    ])

    response_text = response.content.strip()
    classification = parse_classification(response_text)
    explanation = extract_explanation(response_text) if classification in ["GRAVE", "LEVE"] else ""

    # Create new state with consistency classification and explanation
    new_state = state.copy()
    new_state["consistency_classification"] = classification
    new_state["consistency_explanation"] = explanation
    return new_state


def consensus_agent(state: EvaluationState) -> EvaluationState:
    """ConsensusAgent: Combines classifications with algorithmic decision making and detailed error reporting"""

    # Extract classifications
    med_class = state["medication_classification"]
    dosage_class = state["dosage_classification"]
    consistency_class = state["consistency_classification"]

    # Extract explanations
    med_explanation = state.get("medication_explanation", "")
    dosage_explanation = state.get("dosage_explanation", "")
    consistency_explanation = state.get("consistency_explanation", "")

    classifications = [med_class, dosage_class, consistency_class]

    # Apply consensus rules
    if "GRAVE" in classifications:
        final_classification = "GRAVE"
    elif classifications.count("LEVE") >= 2:
        final_classification = "LEVE"
    elif classifications.count("NINGUNA") >= 2:
        final_classification = "NINGUNA"
    else:
        # Default to most severe non-GRAVE classification if tie
        if "LEVE" in classifications:
            final_classification = "LEVE"
        else:
            final_classification = "NINGUNA"

    # Collect error details for GRAVE and LEVE classifications
    error_details = []
    detailed_errors = []

    if med_class == "GRAVE" and med_explanation:
        error_details.append(f"🔴 ERROR CRÍTICO EN MEDICAMENTOS: {med_explanation}")
        detailed_errors.append(f"Medicamentos: {med_explanation}")
    elif med_class == "LEVE" and med_explanation:
        error_details.append(f"🟡 Error menor en medicamentos: {med_explanation}")
        detailed_errors.append(f"Medicamentos: {med_explanation}")

    if dosage_class == "GRAVE" and dosage_explanation:
        error_details.append(f"🔴 ERROR CRÍTICO EN DOSIS: {dosage_explanation}")
        detailed_errors.append(f"Dosis: {dosage_explanation}")
    elif dosage_class == "LEVE" and dosage_explanation:
        error_details.append(f"🟡 Error menor en dosis: {dosage_explanation}")
        detailed_errors.append(f"Dosis: {dosage_explanation}")

    if consistency_class == "GRAVE" and consistency_explanation:
        error_details.append(f"🔴 ERROR CRÍTICO EN COHERENCIA: {consistency_explanation}")
        detailed_errors.append(f"Coherencia: {consistency_explanation}")
    elif consistency_class == "LEVE" and consistency_explanation:
        error_details.append(f"🟡 Error menor en coherencia: {consistency_explanation}")
        detailed_errors.append(f"Coherencia: {consistency_explanation}")

    # Create comprehensive explanation
    explanation = f"""Clasificación final: {final_classification}

Análisis de agentes:
• Medicamentos: {med_class}
• Dosis: {dosage_class}
• Coherencia: {consistency_class}

Reglas aplicadas:
• Si cualquiera es GRAVE → final = GRAVE
• Si la mayoría es LEVE → final = LEVE
• Si la mayoría son NINGUNA → final = NINGUNA"""

    # Add error details if present
    if error_details:
        explanation += f"\n\n⚠️ DETALLES DE ERRORES ENCONTRADOS:\n" + "\n".join(error_details)

    # Add safety recommendation for GRAVE classifications
    if final_classification == "GRAVE":
        explanation += "\n\n🚨 RECOMENDACIÓN: Esta transcripción requiere revisión inmediata por parte de un profesional médico antes de su uso clínico."

    return {
        **state,
        "final_classification": final_classification,
        "consensus_explanation": explanation,
        "error_details": detailed_errors
    }


# Build the graph
def create_medication_evaluation_graph():
    """Create the medication evaluation LangGraph"""

    # Initialize StateGraph
    workflow = StateGraph(EvaluationState)

    # Add nodes (agents)
    workflow.add_node("medication_agent", medication_agent)
    workflow.add_node("dosage_agent", dosage_agent)
    workflow.add_node("consistency_agent", consistency_agent)
    workflow.add_node("consensus_agent", consensus_agent)

    # Define workflow edges
    # Sequential execution to avoid concurrent updates
    workflow.add_edge(START, "medication_agent")
    workflow.add_edge("medication_agent", "dosage_agent")
    workflow.add_edge("dosage_agent", "consistency_agent")
    workflow.add_edge("consistency_agent", "consensus_agent")

    # End the workflow
    workflow.add_edge("consensus_agent", END)

    # Compile the graph
    graph = workflow.compile()

    return graph


# Create the graph instance
medication_evaluation_graph = create_medication_evaluation_graph()


if __name__ == "__main__":
    # Example usage with detailed error reporting
    test_state = {
        "original_text": "El paciente toma Celebrex 200 mg cada 12 horas para el dolor articular. No tiene alergias conocidas.",
        "transcribed_text": "El paciente toma Cerebyx 20 mg cada 8 horas para el dolor articular. Tiene alergias conocidas.",
        "medication_classification": None,
        "dosage_classification": None,
        "consistency_classification": None,
        "final_classification": None,
        "explanations": [],
        "consensus_explanation": "",
        "medication_explanation": "",
        "dosage_explanation": "",
        "consistency_explanation": "",
        "error_details": []
    }

    try:
        result = medication_evaluation_graph.invoke(test_state)

        print("=== RESULTADO DE EVALUACIÓN ===")
        print(f"Texto original: {result['original_text']}")
        print(f"Texto transcrito: {result['transcribed_text']}")
        print(f"\nClasificaciones:")
        print(f"• Medicamentos: {result['medication_classification']}")
        if result.get('medication_explanation'):
            print(f"  └─ {result['medication_explanation']}")
        print(f"• Dosis: {result['dosage_classification']}")
        if result.get('dosage_explanation'):
            print(f"  └─ {result['dosage_explanation']}")
        print(f"• Coherencia: {result['consistency_classification']}")
        if result.get('consistency_explanation'):
            print(f"  └─ {result['consistency_explanation']}")
        print(f"• Final: {result['final_classification']}")
        print(f"\nExplicación del consenso:\n{result['consensus_explanation']}")

        # Show structured error details
        if result.get('error_details'):
            print(f"\n=== ERRORES DETALLADOS ===")
            for i, error in enumerate(result['error_details'], 1):
                print(f"{i}. {error}")

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Necesitas configurar OPENAI_API_KEY para ejecutar las pruebas")
        print("   Ejemplo: export OPENAI_API_KEY='tu_clave_aqui'")