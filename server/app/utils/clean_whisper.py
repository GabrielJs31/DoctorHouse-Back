import re
import json
from typing import Any

def clean_whisper_timestamps(text: str) -> str:
    
    return re.sub(r"\[\d+\.\d+s\s*→\s*\d+\.\d+s\]\s*", "", text)

def clean_json_response(raw: str) -> Any:
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Manejo básico; ajusta si usas otro parser
        return raw