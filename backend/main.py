"""FastAPI 진입점."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import detect as detect_routes
from routes import jobs as jobs_routes
from routes import model as model_routes
from services.inference import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: 모델을 미리 로드해 첫 요청 지연 제거
    logger.info("Loading inference engine…")
    get_engine()
    logger.info("✅ Inference engine ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Fire & Smoke Detection API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_routes.router)
app.include_router(detect_routes.router)
app.include_router(jobs_routes.router)


@app.get("/")
def root():
    return {"name": "Fire & Smoke Detection API", "version": "0.1.0"}
