from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from methods import save_file, transcribe_whisper
from api_IA import extract_data_via_azure
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Audio → Transcripción & Extracción", version="1.0")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/transcribe",
    response_model=Dict[str, Any],
    summary="Graba audio, transcribe y extrae datos clínicos"
)

async def transcribe_and_extract(file: UploadFile = File(...)) -> JSONResponse:
    
    # Validación básica de tipo
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser de tipo audio.")

    # 1) Guardar el archivo
    path = save_file(file)

    # 2) Transcribir con Whisper
    text = transcribe_whisper(path)

    # 3) Enviar texto limpio a Azure y obtener JSON
    data = await extract_data_via_azure(text)

    # 4) Devolver JSON puro al cliente web
    return JSONResponse(content=data)

