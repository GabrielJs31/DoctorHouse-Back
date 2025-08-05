# schemas/transcript.py
from pydantic import BaseModel
from typing import Dict, Any

class ClinicalDataResponse(BaseModel):
    datos_personales: Dict[str, Any]
    motivo_consulta: Dict[str, Any]
