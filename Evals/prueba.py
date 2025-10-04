"""
Prueba del sistema con datos reales de la consulta de cardiología
Este ejemplo muestra un error GRAVE real: antihipertensivo → mercurio
"""

import json
from Evals.medical_metrics import medication_evaluation_graph

def extract_full_dialogue(json_data):
    """Extrae el diálogo completo de la estructura JSON"""
    turns = json_data.get('turns', [])
    dialogue_parts = []
    
    for turn in turns:
        speaker = turn.get('speaker', 'Desconocido')
        text = turn.get('text', '')
        dialogue_parts.append(f"{speaker}: {text}")
    
    return ' '.join(dialogue_parts)

def test_cardiology_dialogue():
    """Test con el diálogo de cardiología real"""
    
    # Simular la carga de los archivos JSON
    original_data = {
  "scenario_id": "cardiologia_01",
  "title": "Consulta de cardiología",
  "context": "Consulta externa de cardiología",
  "turns": [
    {
      "speaker": "Cardiólogo",
      "text": "Buenos días. Pase y siéntese. ¿Qué le trae hoy a la consulta de cardiología?"
    },
    {
      "speaker": "Paciente",
      "text": "Buenos días, doctor. Me dirijo"
    },
    {
      "speaker": "Cardiólogo",
      "text": "mi médico de cabecera porque llevo la tensión alta y últimamente he tenido un par de episodios de dolor en el pecho."
    },
    {
      "speaker": "Paciente",
      "text": "Entiendo. Vamos a hablar de esos dolores. ¿Cuándo empezaron y cómo los describe?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "Pues desde hace unas tres semanas. Me da como una presión"
    },
    {
      "speaker": "Paciente",
      "text": "en el pecho cuando subo escaleras o camino deprisa."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Me dura unos minutos y luego se me pasa."
    },
    {
      "speaker": "Paciente",
      "text": "Ese dolor se acompaña de falta de aire, sudoración o mareo?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "Sí, un poco de falta de aire, pero nada más."
    },
    {
      "speaker": "Paciente",
      "text": "El dolor cede cuando se detiene o se sienta?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "Sí."
    },
    {
      "speaker": "Paciente",
      "text": "al parar mejora. Bien. Eso es importante."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Ahora, respecto a la tensión. ¿Sabe qué cifras ha tenido últimamente?"
    },
    {
      "speaker": "Paciente",
      "text": "En el centro de salud me la tomaron varias veces y siempre estaba sobre 155/95 más o menos."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Vale."
    },
    {
      "speaker": "Paciente",
      "text": "Está tomando"
    },
    {
      "speaker": "Cardiólogo",
      "text": "alguna medicación actualmente?"
    },
    {
      "speaker": "Paciente",
      "text": "Solo paracetamol cuando me duele algo, pero para la tensión no me dieron nada todavía."
    },
    {
      "speaker": "Cardiólogo",
      "text": "De acuerdo. Tiene antecedentes familiares de infarto, angina o ictus?"
    },
    {
      "speaker": "Paciente",
      "text": "Sí, mi padre tuvo un infarto a los 60 años."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Fuma o ha fumado?"
    },
    {
      "speaker": "Paciente",
      "text": "Sí, de joven, pero lo dejé hace 10 años."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Perfecto."
    },
    {
      "speaker": "Paciente",
      "text": "¿Y el colesterol o la glucosa se los han mirado?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "El médico me dijo que el colesterol estaba un poco alto, pero no me dio medicación."
    },
    {
      "speaker": "Paciente",
      "text": "Bien."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Le comento. Por lo que me cuenta,"
    },
    {
      "speaker": "Paciente",
      "text": "podría tratarse de angina de esfuerzo, es decir, dolor torácico al hacer actividad porque al corazón le cuesta recibir suficiente oxígeno."
    },
    {
      "speaker": "Cardiólogo",
      "text": "No es algo para alarmarse en este momento, pero sí debemos estudiarlo y tratar la hipertensión."
    },
    {
      "speaker": "Paciente",
      "text": "Ya, me asusta un poco, la verdad."
    },
    {
      "speaker": "Cardiólogo",
      "text": "normal."
    },
    {
      "speaker": "Paciente",
      "text": "Lo primero es pedirle una prueba de esfuerzo y un electrocardiograma, además de una analítica completa con perfil lipídico."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Mientras tanto, vamos a iniciar tratamiento, ¿de acuerdo? Aquí pone que no tiene alergias."
    },
    {
      "speaker": "Paciente",
      "text": "De acuerdo."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Le voy a recetar un antihipertensivo."
    }
  ]
}
    
    transcribed_data = {
  "scenario_id": "cardiologia_01",
  "title": "Consulta de cardiología",
  "context": "Consulta externa de cardiología",
  "turns": [
    {
      "speaker": "Cardiólogo",
      "text": "Buenos días. Pase y siéntese. ¿Qué le trae hoy a la consulta de cardiología?"
    },
    {
      "speaker": "Paciente",
      "text": "Buenos días, doctor. Me dirijo"
    },
    {
      "speaker": "Cardiólogo",
      "text": "mi médico de cabecera porque llevo la tensión alta y últimamente he tenido un par de episodios de dolor en el pecho."
    },
    {
      "speaker": "Paciente",
      "text": "Entiendo. Vamos a hablar de esos dolores. ¿Cuándo empezaron y cómo los describe?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "Pues desde hace unas tres semanas. Me da como una presión"
    },
    {
      "speaker": "Paciente",
      "text": "en el pecho cuando subo escaleras o camino deprisa."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Me dura unos minutos y luego se me pasa."
    },
    {
      "speaker": "Paciente",
      "text": "Ese dolor se acompaña de falta de aire, sudoración o mareo?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "Sí, un poco de falta de aire, pero nada más."
    },
    {
      "speaker": "Paciente",
      "text": "El dolor cede cuando se detiene o se sienta?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "Sí."
    },
    {
      "speaker": "Paciente",
      "text": "al parar mejora. Bien. Eso es importante."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Ahora, respecto a la tensión. ¿Sabe qué cifras ha tenido últimamente?"
    },
    {
      "speaker": "Paciente",
      "text": "En el centro de salud me la tomaron varias veces y siempre estaba sobre 155/95 más o menos."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Vale."
    },
    {
      "speaker": "Paciente",
      "text": "Está tomando"
    },
    {
      "speaker": "Cardiólogo",
      "text": "alguna medicación actualmente?"
    },
    {
      "speaker": "Paciente",
      "text": "Solo paracetamol cuando me duele algo, pero para la tensión no me dieron nada todavía."
    },
    {
      "speaker": "Cardiólogo",
      "text": "De acuerdo. Tiene antecedentes familiares de infarto, angina o ictus?"
    },
    {
      "speaker": "Paciente",
      "text": "Sí, mi padre tuvo un infarto a los 60 años."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Fuma o ha fumado?"
    },
    {
      "speaker": "Paciente",
      "text": "Sí, de joven, pero lo dejé hace 10 años."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Perfecto."
    },
    {
      "speaker": "Paciente",
      "text": "¿Y el colesterol o la glucosa se los han mirado?"
    },
    {
      "speaker": "Cardiólogo",
      "text": "El médico me dijo que el colesterol estaba un poco alto, pero no me dio medicación."
    },
    {
      "speaker": "Paciente",
      "text": "Bien."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Le comento. Por lo que me cuenta,"
    },
    {
      "speaker": "Paciente",
      "text": "podría tratarse de angina de esfuerzo, es decir, dolor torácico al hacer actividad porque al corazón le cuesta recibir suficiente oxígeno."
    },
    {
      "speaker": "Cardiólogo",
      "text": "No es algo para alarmarse en este momento, pero sí debemos estudiarlo y tratar la hipertensión."
    },
    {
      "speaker": "Paciente",
      "text": "Ya, me asusta un poco, la verdad."
    },
    {
      "speaker": "Cardiólogo",
      "text": "normal."
    },
    {
      "speaker": "Paciente",
      "text": "Lo primero es pedirle una prueba de esfuerzo y un electrocardiograma, además de una analítica completa con perfil lipídico."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Mientras tanto, vamos a iniciar tratamiento, ¿de acuerdo? Aquí pone que tiene alergias."
    },
    {
      "speaker": "Paciente",
      "text": "De acuerdo."
    },
    {
      "speaker": "Cardiólogo",
      "text": "Le voy a recetar un antihipertensivo."
    }
  ]
}
    
    # Para este test específico, vamos a usar solo la parte donde ocurre el error
    original_text = """Consulta de cardiología. Paciente con hipertensión y dolor torácico al esfuerzo. 
    Antecedentes familiares de infarto. Tensión arterial 155/95. 
    Diagnóstico probable: angina de esfuerzo. 
    Plan: prueba de esfuerzo, ECG, analítica con perfil lipídico. 
    Tratamiento: Le voy a recetar un antihipertensivo."""
    
    transcribed_text = """Consulta de cardiología. Paciente con hipertensión y dolor torácico al esfuerzo. 
    Antecedentes familiares de infarto. Tensión arterial 155/95. 
    Diagnóstico probable: angina de esfuerzo. 
    Plan: prueba de esfuerzo, ECG, analítica con perfil lipídico. 
    Tratamiento: Le voy a recetar un gramo de mercurio."""
    
    # Estado inicial
    test_state = {
        "original_text": original_data,
        "transcribed_text": transcribed_data,
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
    
    return test_state

def run_test():
    """Ejecutar la prueba y mostrar resultados"""
    print("=" * 70)
    print("🏥 PRUEBA CON DATOS REALES DE CARDIOLOGÍA")
    print("=" * 70)
    
    test_state = test_cardiology_dialogue()
    
    try:
        result = medication_evaluation_graph.invoke(test_state)
        
        print(f"\n📊 RESULTADOS:")
        print(f"• Medicamentos: {result['medication_classification']}")
        if result.get('medication_explanation'):
            print(f"  └─ {result['medication_explanation']}")
            
        print(f"• Dosis: {result['dosage_classification']}")
        if result.get('dosage_explanation'):
            print(f"  └─ {result['dosage_explanation']}")
            
        print(f"• Coherencia: {result['consistency_classification']}")
        if result.get('consistency_explanation'):
            print(f"  └─ {result['consistency_explanation']}")
            
        print(f"\n🏆 CLASIFICACIÓN FINAL: {result['final_classification']}")
        
        if result.get('error_details'):
            print(f"\n⚠️ ERRORES DETECTADOS:")
            for i, error in enumerate(result['error_details'], 1):
                print(f"{i}. {error}")
                
        print(f"\n📝 EXPLICACIÓN COMPLETA:")
        print(result['consensus_explanation'])
        
        # Verificar que el sistema detectó correctamente el error
        if result['final_classification'] == 'GRAVE':
            print(f"\n✅ ÉXITO: El sistema detectó correctamente el error GRAVE")
        else:
            print(f"\n❌ PROBLEMA: El sistema no detectó el error como GRAVE")
            
    except Exception as e:
        print(f"❌ Error al ejecutar la prueba: {e}")
        print("💡 Verifica que OPENAI_API_KEY esté configurada")

if __name__ == "__main__":
    run_test()