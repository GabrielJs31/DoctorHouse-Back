# utils/file_ops.py
import tempfile
import re
import json
from pathlib import Path
from fastapi import UploadFile
from typing import Any

def save_file(file: UploadFile) -> Path:
    suffix = Path(file.filename).suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    return Path(tmp.name)