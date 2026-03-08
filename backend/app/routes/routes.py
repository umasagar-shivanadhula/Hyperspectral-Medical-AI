"""
Main API Router — connects all route modules.

Note: /health is defined exclusively in backend/main.py.
      It must NOT be duplicated here.
"""
from fastapi import APIRouter
from backend.app.routes.predict import router as predict_router

router = APIRouter()
router.include_router(predict_router, prefix="/predict", tags=["Prediction"])
