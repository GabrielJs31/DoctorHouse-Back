# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.transcribe import router as transcribe_router

def create_app() -> FastAPI:
    app = FastAPI(title="DoctorHouse API")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Registrar rutas
    app.include_router(transcribe_router, prefix="/api/transcribe", tags=["Transcripción"])

    return app

app = create_app()
