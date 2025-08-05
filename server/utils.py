import json
import re
import tempfile
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

def save_file(file: UploadFile) -> Path:
    
    suffix = Path(file.filename).suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)

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

# Extracccion y validacion de datos altura y peso
def calculate_bmi(peso_str: str, altura_str: str) -> float:
    
    # Extraer valores numéricos
    peso_val = re.sub(r"[^0-9.]", "", peso_str)
    altura_val = re.sub(r"[^0-9.]", "", altura_str)
    if not peso_val or not altura_val:
        raise ValueError("Peso y altura deben contener valores numéricos")

    peso = float(peso_val)
    altura = float(altura_val)

    # Normalizar altura en metros si viene en cm
    if altura > 3:
        altura = altura / 100.0

    if altura <= 0:
        raise ValueError("Altura debe ser mayor que cero")

    imc = peso / (altura ** 2)
    return round(imc, 2)

def imc_calculate(data):
    """
    Extrae peso y altura del JSON clínico, calcula el IMC y lo agrega al campo correspondiente.
    Modifica el diccionario data in-place y lo retorna.
    """
    from utils import calculate_bmi  # Importación local para evitar ciclos
    examen = data.get("examen_físico", {})
    peso_str = examen.get("peso_kg", "")
    altura_str = examen.get("altura_cm", "")
    try:
        imc_val = calculate_bmi(peso_str, altura_str)
    except Exception:
        imc_val = None
    if "IMC" not in data:
        data["IMC"] = {"valor": None, "clasificacion": ""}
    data["IMC"]["valor"] = imc_val
    if imc_val is None:
        clasif = "N/A"
    elif imc_val < 18.5:
        clasif = "Bajo peso"
    elif imc_val < 25:
        clasif = "Normal"
    elif imc_val < 30:
        clasif = "Sobrepeso"
    else:
        clasif = "Obesidad"
    data["IMC"]["clasificacion"] = clasif
    return data

