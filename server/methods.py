# methods.py

import json
import re
import time
from pathlib import Path
from typing import Any
from faster_whisper import WhisperModel
from fastapi import UploadFile

# Inicializa el modelo Whisper una sola vez
_model = WhisperModel("small", device="cpu", compute_type="int8")

# Directorio por defecto para guardar audios
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio_files"
AUDIO_DIR.mkdir(exist_ok=True, parents=True)

def save_file(file: UploadFile, directory: Path = AUDIO_DIR) -> Path:
    
    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    path = directory / filename
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path

def transcribe_whisper(path: Path) -> str:
    
    segments, _ = _model.transcribe(str(path), beam_size=5)
    return "".join(seg.text for seg in segments)

def clean_whisper_timestamps(text: str) -> str:
    
    return re.sub(r"\[\d+\.\d+s\s*→\s*\d+\.\d+s\]\s*", "", text)

def clean_json_response(raw: str) -> Any:
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Manejo básico; ajusta si usas otro parser
        return raw
