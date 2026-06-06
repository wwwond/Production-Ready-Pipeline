from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from db.session import create_tables
from api.routers import anomaly


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PRP API 서버 시작 중...")
    create_tables()
    logger.info("PRP API 서버 준비 완료")

    yield  

    logger.info("PRP API 서버 종료")


app = FastAPI(
    title       = "PRP - Precision Realtime Pipeline",
    description = "제조 공정 이상 탐지 시스템 API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anomaly.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "service": "PRP API"}


@app.get("/", tags=["health"])
def root():
    return {
        "service" : "PRP - Precision Realtime Pipeline",
        "version" : "1.0.0",
        "docs"    : "/docs",
    }