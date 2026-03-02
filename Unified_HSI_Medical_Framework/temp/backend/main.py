"""
Unified HSI Medical Framework — FastAPI Backend
"""
import sys
from pathlib import Path

# Ensure project root is on path for ml_pipeline imports
_ROOT = Path(__file__).resolve().parent.parent  # project root (contains ml_pipeline)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router

app = FastAPI(
    title="Unified HSI Medical Framework API",
    description="Spectral-Spatial Multi-Classifier Hyperspectral Medical Image Analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/predict", tags=["Prediction"])


@app.get("/")
def root():
    return {
        "name": "Unified HSI Medical Framework",
        "version": "1.0.0",
        "endpoints": ["/predict/perfusion", "/predict/tumor"],
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
