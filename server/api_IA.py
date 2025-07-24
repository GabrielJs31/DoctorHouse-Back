import os
import time
from typing import Dict, Any

import httpx                   # pip install httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Configuración Azure ──────────────────────────────────────────────────────
AZURE_API_KEY            = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION        = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_API_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT")

# ─── Carga modelo Whisper ────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
AUDIO_DIR  = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Servicio Unificado: Transcribe + Extracción IA")

class AIResponse(BaseModel):
    status: str
    transcription: str | None = None
    extracted_data: Dict[str, Any] | str

async def extract_data_via_azure(txt: str) -> Any:
    prompt = (
        "Eres un extractor de datos. Devuelve un JSON con los campos "
        "\"nombre\", \"fecha\", \"monto\" encontrados en el texto. "
        "Si alguno no existe, pon null.\n\n"
        f"Texto:\n\"\"\"\n{txt}\n\"\"\""
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            AZURE_API_ENDPOINT,
            params={"api-version": AZURE_API_VERSION},
            headers=headers,
            json=payload
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Error IA ({resp.status_code}): {resp.text}"
        )

    result = resp.json()
    # La respuesta de Azure suele venir como cadena JSON; 
    # si quieres devolverla ya parseada:
    try:
        return result["choices"][0]["message"]["content"]
    except Exception:
        return result

@app.post("/transcribe", response_model=AIResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    # Validar tipo de archivo
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Solo archivos de audio permitidos")
    # Guardar temporalmente
    file_path = AUDIO_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Transcribir
    start = time.time()
    segments, info = whisper_model.transcribe(
        str(file_path),
        beam_size=1,
        vad_filter=True
    )
    transcription = "\n".join(
        f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}" for s in segments
    )
    elapsed = time.time() - start

    # Extraer datos usando Azure
    extracted = await extract_data_via_azure(transcription)

    return AIResponse(
        status="ok",
        transcription=transcription,
        extracted_data=extracted
    )

@app.post("/upload_txt", response_model=AIResponse)
async def upload_txt(file: UploadFile = File(...)):
    if file.content_type != "text/plain":
        raise HTTPException(400, "Solo archivos .txt permitidos")
    txt = (await file.read()).decode("utf-8", errors="strict")
    extracted = await extract_data_via_azure(txt)

    return AIResponse(
        status="ok",
        extracted_data=extracted
    )

@app.get("/health")
def health():
    return {"status": "alive"}

if __name__ == "__main__":
    # Comando para arrancar:
    # uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
