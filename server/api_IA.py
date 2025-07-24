import os
from typing import Dict, Any
import httpx                    # pip install httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai                   # pip install openai
import uvicorn                  # pip install uvicorn[standard]
from pathlib import Path
from dotenv import load_dotenv 

load_dotenv()
AZURE_API_KEY            = os.getenv("AZURE_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") 
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")

openai.api_key     = AZURE_API_KEY
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_endpoint = AZURE_OPENAI_ENDPOINT

class AIResponse(BaseModel):
    status: str
    extracted_data: Dict[str, Any] | str


async def extract_data_via_azure(txt: str) -> str:
    prompt = (
        "Eres un extractor de datos. Devuelve un JSON con los campos "
        "\"nombre\", \"fecha\", \"monto\" encontrados en el texto. "
        "Si alguno no existe, pon null.\n\n"
        f"Texto:\n\"\"\"\n{txt}\n\"\"\""
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            AZURE_OPENAI_ENDPOINT,
            params={"api-version": AZURE_OPENAI_API_VERSION},
            headers=headers,
            json=payload
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Error IA ({resp.status_code}): {resp.text}"
        )

    result = resp.json()
    return result["choices"][0]["message"]["content"]


# async def send_to_ia_endpoint(data: AIResponse) -> None:
   
#     async with httpx.AsyncClient(timeout=15) as client:
#         resp = await client.post(IA_ENDPOINT, json=data.model_dump())
#         if resp.status_code >= 400:
#             raise HTTPException(
#                 status_code=502,
#                 detail=f"Fallo al reenviar a /IA: {resp.text}"
#             )


app = FastAPI(title="Servicio de Ingesta .txt y Extracción IA (Azure)")

@app.post("/upload_txt")
async def upload_txt(file: UploadFile = File(...)):
    if file.content_type != "text/plain":
        raise HTTPException(400, "Solo archivos .txt permitidos")
    txt = (await file.read()).decode("utf-8", errors="strict")

    extracted = await extract_data_via_azure(txt)
    if not extracted:
        raise HTTPException(500, "No se pudo extraer información")

    return JSONResponse(
        content=AIResponse(status="ok", extracted_data=extracted).model_dump()
    )

@app.get("/health")
def health():
    return {"status": "alive"}


if __name__ == "__main__":
    uvicorn.run("api_IA:app", host="0.0.0.0", port=8000, reload=True)
