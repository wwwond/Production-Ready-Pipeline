"""
api/main.py
===========
역할
----
FastAPI 애플리케이션의 진입점(Entry Point)입니다.
앱 초기화, 라우터 등록, 미들웨어 설정을 담당합니다.

FastAPI란?
----------
Python으로 REST API를 빠르게 만들 수 있는 웹 프레임워크입니다.
자동으로 Swagger 문서를 생성해주고, Pydantic으로 데이터 검증을 처리합니다.

  실행 후 접속:
    API 문서 (Swagger) : http://localhost:8000/docs
    API 문서 (ReDoc)   : http://localhost:8000/redoc
    헬스체크           : http://localhost:8000/health

앱 시작 시 동작
---------------
1. DB 테이블 자동 생성 (없으면 만들고, 있으면 스킵)
2. 라우터 등록 (/api/v1/anomaly/...)
3. CORS 설정 (Grafana 대시보드에서 API 호출 허용)

실행 방법
---------
  # 로컬 개발
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

  # Docker
  docker compose up -d
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from db.session import create_tables
from api.routers import anomaly


# ── 앱 시작/종료 이벤트 ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 시작 시 DB 테이블을 자동 생성합니다.
    앱이 종료될 때 정리 작업을 수행합니다.
    """
    # 시작 시
    logger.info("PRP API 서버 시작 중...")
    create_tables()
    logger.info("PRP API 서버 준비 완료")

    yield  # 앱 실행 중

    # 종료 시
    logger.info("PRP API 서버 종료")


# ── FastAPI 앱 초기화 ─────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PRP - Precision Realtime Pipeline",
    description = "제조 공정 이상 탐지 시스템 API",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── CORS 설정 ─────────────────────────────────────────────────────────────────
# Grafana 대시보드에서 이 API를 호출할 수 있도록 허용합니다.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 프로덕션에서는 특정 도메인만 허용으로 변경
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 라우터 등록 ───────────────────────────────────────────────────────────────
# anomaly.py의 엔드포인트들을 /api/v1 prefix로 등록합니다.
#
# 등록 결과:
#   POST /api/v1/anomaly/predict
#   GET  /api/v1/anomaly/results
#   GET  /api/v1/anomaly/results/{id}
#   GET  /api/v1/anomaly/stats
#   GET  /api/v1/anomaly/heatmap/{id}

app.include_router(anomaly.router, prefix="/api/v1")


# ── 헬스체크 ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
def health_check():
    """
    서버 상태를 확인하는 엔드포인트입니다.
    Docker Compose의 healthcheck에서 사용합니다.
    """
    return {"status": "ok", "service": "PRP API"}


@app.get("/", tags=["health"])
def root():
    return {
        "service" : "PRP - Precision Realtime Pipeline",
        "version" : "1.0.0",
        "docs"    : "/docs",
    }
