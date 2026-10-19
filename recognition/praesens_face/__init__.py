"""Praesens face recognition: InsightFace + FAISS, packaged for a FastAPI app."""

from .config import Settings
from .engine import EnrollmentError, FaceEngine, FaceResult, GuestEvent
from .router import build_router

__all__ = ["Settings", "FaceEngine", "FaceResult", "GuestEvent", "EnrollmentError", "build_router"]
