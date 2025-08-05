# routers/transcribe.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.clinical_extractor import ClinicalExtractorService
from schemas.transcript import ClinicalDataResponse

router = APIRouter()

# Instancia del servicio (puede inyectarse)
extractor = ClinicalExtractorService()

@router.post("/", response_model=ClinicalDataResponse)
async def transcribe_and_extract(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Sólo archivos de tipo audio.")
    result = await extractor.process_audio(file)
    return JSONResponse(content=result)
