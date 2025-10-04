import argparse
import random
from pydub import AudioSegment
import sys
import os


## Ejemplo de como usar:
# python degradar_audio.py original.wav malo.mp3

# Ejemplo de como usar en el proyecto:
#python modify_audio.py generated_datasets/dialogues/dialogue1.wav generated_datasets/dialogues_modify/dialogue1.modify.mp3
"""
original.wav	Archivo de entrada (obligatorio).	mi_voz.wav
malo.mp3	Archivo de salida (obligatorio).	grabacion_vieja.mp3
-r o --rate	Tasa de muestreo (Hz).	// valores aleatorios entre 8000 y 6000
-w o --width	Profundidad de bits (1 = 8-bit).	// Siempre 1
-g o --gain	Aumento de volumen (dB).	// Valores aleatorios entre 25 y 35
-b o --bitrate	Bitrate para MP3.	// Valores aleatorios entre 32k
"""
def degradar_audio(input_file, output_file):
    """
    Carga un archivo de audio y aplica una degradación con valores aleatorios.
    """
    try:
        # 1. Generar Parámetros de Degradación Aleatorios
        
        # -r (Tasa de Muestreo): Entre 5000 y 8000 Hz
        rate = random.randint(5000, 8000)
        
        # -w (Profundidad de Bits): Siempre 1 (8-bit)
        width = 1 
        
        # -g (Ganancia/Distorsión): Entre 20 y 30 dB
        gain = random.randint(15, 20)
        
        # -b (Bitrate): Entre 16 y 32 kbps (solo para MP3)
        #bitrate_num = random.randint(32)
        bitrate_num = 64
        bitrate = f"{bitrate_num}k"
        
        # 2. Cargar el archivo de audio
        audio = AudioSegment.from_file(input_file)
        print(f"✅ Archivo '{input_file}' cargado con éxito.")
        
        # 3. Aplicar degradación
        
        # Reducir la Tasa de Muestreo
        audio_degradado = audio.set_frame_rate(rate)
        
        # Reducir la Profundidad de Bits (8-bit)
        audio_degradado = audio_degradado.set_sample_width(width)
        
        # Aumentar el volumen para causar distorsión (clipping)
        audio_degradado = audio_degradado + gain
        
        print("\n--- Parámetros Aleatorios Generados ---")
        print(f"⚙️ Tasa de Muestreo (-r): {rate} Hz (Degradación de agudos)")
        print(f"⚙️ Profundidad de Bits (-w): {width} (Ruido de cuantificación)")
        print(f"⚙️ Ganancia/Distorsión (-g): {gain} dB (Distorsión/Clipping)")
        print(f"⚙️ Bitrate (-b): {bitrate} (Artefactos de compresión)")
        print("--------------------------------------\n")


        # 4. Exportar el archivo degradado
        formato = output_file.split('.')[-1]
        
        if formato == 'mp3':
            audio_degradado.export(
                output_file,
                format=formato,
                bitrate=bitrate
            )
        else: # Para otros formatos (wav, ogg, etc.)
            # Para formatos sin pérdida como WAV, solo se aplicarán la Tasa de Muestreo, el Ancho de Bits y la Ganancia
            audio_degradado.export(
                output_file,
                format=formato
            )

        print(f"🎉 Éxito: El archivo degradado se guardó como '{output_file}'.")

    except FileNotFoundError:
        print(f"❌ Error: El archivo de entrada '{input_file}' no fue encontrado.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ocurrió un error: {e}", file=sys.stderr)
        sys.exit(1)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Degrada la calidad de un archivo de audio con parámetros aleatorios."
    )
    
    # Argumentos obligatorios: Solo el archivo de entrada y salida
    parser.add_argument("input_file", help="Ruta al archivo de audio de entrada (ej: original.wav).")
    parser.add_argument("output_file", help="Ruta al archivo de audio de salida (ej: malo_random.mp3).")
    
    # Ya no se necesitan los argumentos opcionales, ya que los valores se generan aleatoriamente
    args = parser.parse_args()

    # La función degradar_audio ya no recibe los parámetros de degradación
    degradar_audio(args.input_file, args.output_file)"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesa por lotes una serie de diálogos con degradación aleatoria."
    )
    
    # Ahora el script espera un inicio y un fin, no rutas de archivo.
    parser.add_argument("start_num", type=int, help="Número inicial de la serie de diálogos (ej: 1 para dialogue1.wav).")
    parser.add_argument("end_num", type=int, help="Número final de la serie de diálogos (ej: 10 para dialogue10.wav).")
    
    args = parser.parse_args()
    
    # Definición de las carpetas base (basado en tu ejemplo)
    INPUT_DIR = "generated_datasets/dialogues"
    OUTPUT_DIR = "generated_datasets/dialogues_modify"
    
    # 1. Crear la carpeta de salida si no existe
    # 'exist_ok=True' previene errores si la carpeta ya existe.
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"📁 Directorio de salida verificado: '{OUTPUT_DIR}'")
    except Exception as e:
        print(f"❌ Error al crear el directorio de salida: {e}")
        sys.exit(1)

    if args.start_num > args.end_num:
        print("❌ Error: El número inicial debe ser menor o igual que el número final.")
        sys.exit(1)

    print(f"\n🤖 Iniciando procesamiento de diálogo {args.start_num} a {args.end_num}...")

    # 2. Iterar sobre el rango de números
    for i in range(args.start_num, args.end_num + 1):
        # 3. Construir las rutas completas para cada archivo
        
        # Entrada: generated_datasets/dialogues/dialogueN.wav
        input_filename = f"dialogue{i}.wav"
        input_path = os.path.join(INPUT_DIR, input_filename)
        
        # Salida: generated_datasets/dialogues_modify/dialogueN.modify.mp3
        output_filename = f"dialogue{i}.modify.mp3"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n\n================================================")
        print(f"          PROCESANDO: DIÁLOGO {i} / {args.end_num}")
        print(f"================================================")
        
        # 4. Llamar a la función de degradación
        degradar_audio(input_path, output_path)