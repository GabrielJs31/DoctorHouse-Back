# server/app/services/clinical_extractor.py
import os
import re
import json
import httpx
from typing import Dict, Any
from fastapi import UploadFile, HTTPException
from dotenv import load_dotenv, find_dotenv
from utils.clean_whisper import clean_whisper_timestamps
from utils.save_files import save_file
from utils.whisper_service import transcribe_whisper
from utils.imc import imc_calculate,calculate_bmi

# Carga variables de entorno
load_dotenv(find_dotenv())

class ClinicalExtractorService:
    def __init__(self):
        self.api_key = os.getenv("AZURE_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not all([self.api_key, self.api_version, self.endpoint]):
            raise RuntimeError("Faltan variables de entorno de Azure OpenAI")

    async def extract_data(self, txt: str) -> Dict[str, Any]:
        """
        Construye el prompt, llama a Azure OpenAI y parsea la respuesta JSON.
        """
        # 1) Limpia la transcripción
        clean_txt = clean_whisper_timestamps(txt)

        # 2) Prompt intro y estructura esperada
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
            "11.Si el peso o altura se dicen en libras, metros o pies, convertir a kg y cm respectivamente\n" 
            "12. Los campos de datos_personales siempre deben estar completos, no dejar campos vacíos\n"
        )
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

        #Construccion del prompt para Azure OpenAI
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
            "Content-Type": "application/json", "api-key": self.api_key
        }

        params = {"api-version": self.api_version}

        #Llamada a Azure OpenAI
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.endpoint,
                params=params,
                headers=headers,
                json=payload
            )
        if resp.status_code != 200:
            raise HTTPException(502, f"Error IA {resp.status_code}: {resp.text}")

        #Extrae el texto bruto y limpia fences Markdown
        raw = resp.json().get("choices", [])[0].get("message", {}).get("content", "")
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.IGNORECASE).strip()

        #Extrae solo el primer objeto JSON
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(cleaned)
        except json.JSONDecodeError as e:
            raise HTTPException(
                502,
                f"Error parseando JSON de IA: {e}\nRespuesta cruda:\n{repr(cleaned)}"
            )

        return obj

    async def process_audio(self, file: UploadFile) -> Dict[str, Any]:
        """
        Flujo completo: guardar, transcribir, extraer datos y calcular IMC.
        """
        # Guardar archivo temporal
        path = save_file(file)
        # Transcripción
        texto = transcribe_whisper(path)
        # Extracción de datos clínicos
        data = await self.extract_data(texto)
        #Extraer peso y altura, calcular IMC
        data = calculate_bmi(data)
        # Cálculo y anexado de IMC
        data = imc_calculate(data)
        return data
