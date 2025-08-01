from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from methods import save_file, transcribe_whisper,calculate_bmi
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

@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_and_extract(file: UploadFile = File(...)) -> JSONResponse:
    # 1) Validación
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Sólo archivos de tipo audio.")

    # 2) Guardar y transcribir
    path = save_file(file)
    texto = transcribe_whisper(path)

    # 3) Extraer JSON clínico de Azure
    data = await extract_data_via_azure(texto)

    # 4) Extraer peso y altura del JSON
    #    Asumimos que vienen en 'examen_físico'
    examen = data.get("examen_físico", {})
    peso_str   = examen.get("peso_kg", "")
    altura_str = examen.get("altura_cm", "")

    # 5) Calcular IMC
    try:
        imc_val = calculate_bmi(peso_str, altura_str)
    except Exception as e:
        # Si falla el parseo, dejamos el valor en N/A
        imc_val = None

    # 6) Rellenar campo 'valor' en data["IMC"]
    if "IMC" not in data:
        data["IMC"] = {"valor": None, "clasificacion": ""}
    data["IMC"]["valor"] = imc_val

    # 7) Obtener clasificación del IMC
    #    Aquí un ejemplo local sencillo:
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

    # 8) Devolver el JSON completo al cliente
    return JSONResponse(content=data)

