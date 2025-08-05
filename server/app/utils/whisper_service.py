# utils/whisper_service.py
from faster_whisper import WhisperModel
from pathlib import Path

_model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_whisper(path: Path) -> str:
    segments, _ = _model.transcribe(str(path), beam_size=5)
    return "".join(seg.text for seg in segments)
