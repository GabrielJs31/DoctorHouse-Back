# utils/imc.py
import re
from typing import Dict, Any

def calculate_bmi(peso_str: str, altura_str: str) -> float:
    """
    Extrae valores numéricos de peso (kg) y altura (cm o m), 
    normaliza altura > 3 como cm, y devuelve IMC redondeado.
    """
    peso_val = re.sub(r"[^0-9.]", "", peso_str)
    altura_val = re.sub(r"[^0-9.]", "", altura_str)
    if not peso_val or not altura_val:
        raise ValueError("Peso y altura deben contener valores numéricos")

    peso = float(peso_val)
    altura = float(altura_val)
    # Si altura en cm, convertir a metros
    if altura > 3:
        altura /= 100.0
    if altura <= 0:
        raise ValueError("Altura debe ser mayor que cero")

    imc = peso / (altura ** 2)
    return round(imc, 2)


def calculate_and_append_imc(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lee 'peso_kg' y 'altura_cm' de data["examen_físico"], calcula IMC,
    lo inserta en data["IMC"] con 'valor' y 'clasificacion', y retorna data.
    """
    examen = data.get("examen_físico", {})
    peso_str = examen.get("peso_kg", "")
    altura_str = examen.get("altura_cm", "")

    try:
        imc_val = calculate_bmi(peso_str, altura_str)
    except Exception:
        imc_val = None

    # Determinar clasificación
    if imc_val is None:
        clasif = "N/A"
    elif imc_val < 18.5:
        clasif = "Bajo peso"
    elif imc_val < 25:
        clasif = "Peso normal"
    elif imc_val < 30:
        clasif = "Sobrepeso"
    elif imc_val < 35:
        clasif = "Obesidad grado I"
    elif imc_val < 40:
        clasif = "Obesidad grado II"
    else:
        clasif = "Obesidad grado III (mórbida)"

    # Insertar/actualizar sección IMC
    data.setdefault("IMC", {})
    data["IMC"]["valor"] = imc_val
    data["IMC"]["clasificacion"] = clasif

    return data
