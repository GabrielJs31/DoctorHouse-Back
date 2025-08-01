import os
import re
import json
import httpx
from dotenv import load_dotenv, find_dotenv
from fastapi import HTTPException
from typing import Dict, Any
from methods import clean_whisper_timestamps


load_dotenv(find_dotenv())

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_API_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")


async def extract_data_via_azure(txt: str) -> Dict[str, Any]:
    
    # 2) Limpia la transcripción
    clean_txt = clean_whisper_timestamps(txt)
    prompt_intro = (
        "Extrae todos los campos clínicos del texto y devuelve SOLO JSON válido. Reglas CRÍTICAS:\n"
        "1. Los campos de datos_personales siempre debes enviarme llenos sin excepciones\n"
        "2. Para 'fecha_nacimiento': extraer SOLO fecha en formato DD/MM/AAAA\n"
        "3. Para 'enfermedad_actual': siempre debe estar enfermedad actual con sus tratamientos, incluido al especialista que se derivara el caso si lo requiere y examenes requeridos\n"
        "4. Para 'posibles_enfermedades': extraer SOLO si existen posibles enfermedades poner tambien el especialista al que se derivara si se lo requiere SI NO existen posibles_enfermedades entonces poner 'No se encontraron otras posibles enfermedades' \n"
        "5. Para posibles_enfermedades: seguir agregando posibles_enfermedades hasta un máximo de 2, si hay más de 2, solo extraer las 2 más relevantes\n"
        "6. Para enfermedad_actual y posibles_enfermedades: generar tratamiento, examenes requeridos, derivacion a especialista y recomendaciones en base a examen_fisico y motivo_consulta SOLO si no encuentras en el texto\n"
        "7. Para derivacion_especialista: SI NO es necesario un especialista solo poner 'Medico General'\n"
        "8. Para campos numéricos: solo el valor numérico\n"
        "9. Si un dato no existe: usar 'N/A'\n"
        "10. Estructura EXACTA requerida:\n"
        "11. Los campos de datos_personales siempre deben estar completos, no dejar campos vacíos\n"
    )
    # 3) Estructura detallada (original ampliada)
    expected = {
           "datos_personales": {
                "nombre": "",
                "apellido": "",
                "cédula": "",
                "sexo": "",
                "tipo_sangre": "",
                "fecha_nacimiento": "",
                "edad": "",
                "teléfono": "",
                "móvil": "",
                "fecha_consulta": ""
            },
            "motivo_consulta": {
                "motivo": "",
                "lugar": ""
            },
            "enfermedad_actual": {
                "descripción": "",
                "tratamiento": "",
                "examenes_requeridos": "",
                "derivacion_especialista": "",
                "recomendaciones": ""
            },
 
            "posibles_enfermedades": {
                "posible_enfermedad 1": {
                    "description": "",
                    "tratamiento": "",
                    "examenes_requeridos": "",
                    "derivacion_especialista": "",
                    "recomendaciones": ""
                },
                "posible_enfermedad 2": {
                    "description": "",
                    "tratamiento": "",
                    "examenes_requeridos": "",
                    "derivacion_especialista": "",
                    "recomendaciones": ""
                },
            },
            "antecedentes": {
                "personales": "",
                "alergias": "",
                "medicamentos": "",
                "problemas_cardiovasculares": "",
                "fuma": "",
                "familiares": "",
                "intervenciones_quirúrgicas": "",
                "problemas_coagulación": "",
                "problemas_anestésicos": "",
                "alcohol": ""
            },
            "signos_vitales": {
                
                "saturación_oxígeno": "",
                "frecuencia_respiratoria": "",
                "frecuencia_cardíaca": "",
                "presión_arterial": "",
                "temperatura_c": ""
            },
            "examen_físico": {
                "altura_cm": "",
                "peso_kg": "",
                "cabeza_cuello": "",
                "tórax": "",
                "rscs": "",
                "abdomen": "",
                "extremidades": ""
            },
            "diagnóstico_tratamiento": {
                "diagnóstico_presuntivo": "",
                "tratamiento": ""
            },
            "IMC": {
                "valor": "",
                "clasificacion": ""
            }
            

        }

        
    # 4) Prompt: fuerza un único JSON
    system_prompt = (
        "Eres un experto en medicina y extracción de datos clínicos. "
        "Tu tarea es extraer información de historias clínicas.\n\n"
        f"{prompt_intro}\n"
        f"Estructura EXACTA requerida:\n{json.dumps(expected, ensure_ascii=False)}\n\n"
        f"Texto a procesar:\n\"\"\"\n{clean_txt}\n\"\"\""
        
        f"{json.dumps(expected, ensure_ascii=False)}"
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": clean_txt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }

    # 5) Llamada a Azure OpenAI
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            AZURE_API_ENDPOINT,
            params={"api-version": AZURE_API_VERSION},
            headers=headers,
            json=payload
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"Error IA {resp.status_code}: {resp.text}")

    # 6) Extrae el texto bruto y limpia fences Markdown
    raw = resp.json().get("choices", [])[0].get("message", {}).get("content", "")
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.IGNORECASE).strip()

    # 7) Extrae solo el primer objeto JSON
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(
            502,
            f"Error parseando JSON de IA: {e}\nRespuesta cruda:\n{repr(cleaned)}"
        )

    return obj
