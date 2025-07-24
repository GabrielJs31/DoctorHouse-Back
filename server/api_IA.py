import json
import os
import re
import time
from typing import Dict, Any

import httpx  # pip install httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel  # pip install faster-whisper
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
import openai  # pip install openai
import uvicorn




# Carga variables de entorno
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Función para validar entornos
def get_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"La variable de entorno '{var_name}' no está definida.")
    return value

# Configuración Azure
AZURE_API_KEY      = get_env("AZURE_API_KEY")
AZURE_API_VERSION  = get_env("AZURE_OPENAI_API_VERSION")
AZURE_API_ENDPOINT = get_env("AZURE_OPENAI_ENDPOINT")

# (opcional) Configuración cliente OpenAI, no usado en httpx pero por consistencia
openai.api_key     = AZURE_API_KEY
openai.api_version = AZURE_API_VERSION
openai.api_base    = AZURE_API_ENDPOINT

# Prepara carpeta de audio
audio_dir = Path(__file__).parent / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

# Carga modelo Whisper
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# Inicializa FastAPI
app = FastAPI(title="Servicio Unificado: Transcribe + Extracción Historia Clínica IA")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIResponse(BaseModel):
    status: str
    extracted_data: Dict[str, Any]


def clean_json_response(raw: str) -> Dict[str, Any]:
    # Limpia posibles bloques de markdown y parsea JSON
    cleaned = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parseando JSON de IA: {e}")


async def extract_data_via_azure(txt: str) -> Dict[str, Any]:
    # Construye prompt con todos los campos requeridos
    prompt_intro = (
        "Extrae todos los campos de la historia clínica del siguiente texto y organízalos en formato JSON. "
        "Si algún dato falta, devuelve 'N/A'. Los datos deben ser solo el valor numérico o texto. "
        "Ignora cualquier otro texto que no corresponda a los campos."
    )
    fields_list = (
        "### 1. DATOS PERSONALES\n"
        "- nombre\n- apellido\n- cédula\n- sexo\n- tipo_sangre\n- fecha_nacimiento\n"
        "- edad\n- teléfono\n- móvil\n- fecha_consulta\n"
        "### 2. MOTIVO DE CONSULTA\n- motivo\n- lugar\n"
        "### 3. ENFERMEDAD ACTUAL\n- descripción\n"
        "### 4. ANTECEDENTES\n- personales\n- alergias\n- medicamentos\n"
        "- problemas_cardiovasculares\n- fuma\n- familiares\n"
        "- intervenciones_quirúrgicas\n- problemas_coagulación\n- problemas_anestésicos\n- alcohol\n"
        "### 5. SIGNOS VITALES\n- peso_kg\n- saturación_oxígeno\n- frecuencia_respiratoria\n"
        "- frecuencia_cardíaca\n- presión_arterial\n- temperatura_c\n"
        "### 6. EXAMEN FÍSICO\n- cabeza_cuello\n- tórax\n- rscs\n- abdomen\n- extremidades\n"
        "### 7. DIAGNÓSTICO Y TRATAMIENTO\n- diagnóstico_presuntivo\n- tratamiento"
    )
    prompt = (
        f"{prompt_intro}\n\n{fields_list}\n"
        f"\nTexto:\n\"\"\"\n{txt}\n\"\"\""
    )

    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            AZURE_API_ENDPOINT,
            params={"api-version": AZURE_API_VERSION},
            headers=headers,
            json=payload
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error IA ({resp.status_code}): {resp.text}")

    data = resp.json()
    raw = data.get("choices", [])[0].get("message", {}).get("content", "")
    if not raw or not raw.strip():
        raise HTTPException(status_code=502, detail="Respuesta vacía de IA: no hay contenido para parsear")
    return clean_json_response(raw)


@app.post("/transcribe", response_model=AIResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    # Aceptar audio wav, mp3, ogg
    filename = file.filename.lower()
    if not (file.content_type.startswith("audio/") or filename.endswith(('.wav', '.mp3', '.ogg'))):
        raise HTTPException(status_code=400, detail="Solo archivos de audio permitidos (.wav, .mp3, .ogg)")

    # Guarda audio en carpeta `audio`
    audio_path = audio_dir / file.filename
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Transcribe audio
    start = time.time()
    segments, _ = whisper_model.transcribe(str(audio_path), beam_size=1, vad_filter=True)
    transcription = "\n".join(f"[{s.start:.2f}s → {s.end:.2f}s] {s.text}" for s in segments)
    print(f"Transcripción en {time.time() - start:.1f}s")

    # Guarda la transcripción en un archivo .txt
    txt_path = audio_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcription)

    # Extrae datos con Azure
    extracted = await extract_data_via_azure(transcription)

    # Devuelve solo el JSON con extracted_data
    return JSONResponse(content={"status": "ok", "extracted_data": extracted})


@app.post("/upload_txt", response_model=AIResponse)
async def upload_txt(file: UploadFile = File(...)):
    if file.content_type != "text/plain":
        raise HTTPException(status_code=400, detail="Solo archivos .txt permitidos")
    txt = (await file.read()).decode("utf-8", errors="strict")
    extracted = await extract_data_via_azure(txt)
    return JSONResponse(content={"status": "ok", "extracted_data": extracted})


@app.get("/health")
def health():
    return {"status": "alive"}


if __name__ == "__main__":
    uvicorn.run("api_IA:app", host="0.0.0.0", port=8000, reload=True)
