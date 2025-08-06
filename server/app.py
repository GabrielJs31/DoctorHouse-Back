from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.services.clinical_extractor import extract_data_via_azure
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from utils import imc_calculate, save_file, transcribe_whisper
from logs import Logs
logs = Logs()

app = FastAPI()

# Modelo de datos de entrada
class TranscriptRequest(BaseModel):
    txt: str

# Modelo de datos de salida
class ExtractedDataResponse(BaseModel):
    data: Dict[str, Any]

    
@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_and_extract(file: UploadFile = File(...)) -> JSONResponse:
    #Validación archivos tipo audio
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Sólo archivos de tipo audio.")
    #Guardar y transcribir
    path = save_file(file)
    logs.write(f"Archivo guardado: {path}")
    texto = transcribe_whisper(path)
    logs.write(f"Transcripción realizada para: {path}")
    # Extraer JSON clínico 
    data = await extract_data_via_azure(texto)
    logs.write(f"Extracción clínica completada para: {path}")
    #IMC
    data = imc_calculate(data)
    logs.write(f"IMC calculado para: {path}")
    #Devolver el JSON completo al cliente
    return JSONResponse(content=data)
