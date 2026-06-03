"""
api/routers/anomaly.py
======================
역할
----
이상 탐지와 관련된 FastAPI 엔드포인트를 정의하는 라우터 파일입니다.

라우터란?
---------
FastAPI에서 관련된 엔드포인트들을 묶어서 관리하는 단위입니다.
main.py에서 이 라우터를 등록하면 엔드포인트가 활성화됩니다.

  main.py  →  app.include_router(router)  →  /api/v1/anomaly/...

엔드포인트 목록
---------------
1. POST /predict          : 이미지 경로를 받아 추론 실행
2. GET  /results          : 전체 추론 결과 목록 조회
3. GET  /results/{id}     : 특정 결과 단건 조회
4. GET  /stats            : 모델별 통계 조회 (Grafana용)
5. GET  /heatmap/{id}     : 히트맵 이미지 파일 반환

왜 이 구조인가?
--------------
Kafka Consumer가 추론 후 결과를 DB에 저장하고,
FastAPI는 저장된 결과를 조회하는 역할을 담당합니다.
POST /predict는 Kafka 없이 직접 추론할 때 사용합니다. (개발/테스트용)
"""

import yaml
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from db.session import get_db
from db.models import AnomalyResult
from api.schemas import AnomalyResultResponse, AnomalyResultList, StatsResponse
from models.autoencoder.inference import AnomalyDetector
from utils.notify import send_slack_alert, should_alert

router = APIRouter(prefix="/anomaly", tags=["anomaly"])

# config 로드 (전역 1회)
with open("config/config.yaml", "r") as f:
    _config = yaml.safe_load(f)


def _get_detector() -> AnomalyDetector:
    """
    AnomalyDetector 인스턴스를 반환합니다.
    매 요청마다 새로 만들지 않고 재사용합니다.
    """
    return AnomalyDetector(_config)


# ── 1. 직접 추론 (개발/테스트용) ──────────────────────────────────────────────

@router.post("/predict", response_model=AnomalyResultResponse)
def predict(image_path: str, db: Session = Depends(get_db)):
    """
    이미지 경로를 받아 즉시 추론하고 결과를 DB에 저장합니다.
    Kafka 없이 단일 이미지를 테스트할 때 사용합니다.

    Args:
        image_path: 추론할 이미지 경로 (예: data/mvtec/bottle/test/good/000.png)

    Returns:
        추론 결과 (AnomalyResultResponse)
    """
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {image_path}")

    # 추론 실행
    detector = _get_detector()
    result   = detector.predict(image_path)

    # DB 저장
    db_result = AnomalyResult(
        image_path    = result["image_path"],
        model_type    = _config["model"]["current"],
        anomaly_score = result["score"],
        is_anomaly    = result["is_anomaly"],
        heatmap_path  = result["heatmap_path"],
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    # Slack 알림 (alert_threshold 이상일 때만)
    if should_alert(result["score"], _config["slack"]["alert_threshold"]):
        send_slack_alert(
            image_path   = result["image_path"],
            score        = result["score"],
            is_anomaly   = result["is_anomaly"],
            heatmap_path = result["heatmap_path"],
            threshold    = _config["model"]["threshold"],
        )

    logger.info(f"추론 완료 | id={db_result.id} | score={result['score']:.6f}")
    return db_result


# ── 2. 결과 목록 조회 ─────────────────────────────────────────────────────────

@router.get("/results", response_model=AnomalyResultList)
def get_results(
    model_type: str = None,   # 'autoencoder' or 'patchcore' 필터링
    is_anomaly: bool = None,  # True/False 필터링
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    추론 결과 목록을 조회합니다.
    model_type, is_anomaly로 필터링 가능합니다.

    사용 예시:
        GET /anomaly/results                          → 전체 조회
        GET /anomaly/results?model_type=autoencoder   → AutoEncoder 결과만
        GET /anomaly/results?is_anomaly=true          → 이상 탐지된 것만
    """
    query = db.query(AnomalyResult)

    if model_type:
        query = query.filter(AnomalyResult.model_type == model_type)
    if is_anomaly is not None:
        query = query.filter(AnomalyResult.is_anomaly == is_anomaly)

    total   = query.count()
    results = query.order_by(AnomalyResult.created_at.desc()).offset(offset).limit(limit).all()

    return AnomalyResultList(total=total, results=results)


# ── 3. 결과 단건 조회 ─────────────────────────────────────────────────────────

@router.get("/results/{result_id}", response_model=AnomalyResultResponse)
def get_result(result_id: int, db: Session = Depends(get_db)):
    """
    특정 ID의 추론 결과를 조회합니다.
    """
    result = db.query(AnomalyResult).filter(AnomalyResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail=f"결과를 찾을 수 없습니다: id={result_id}")
    return result


# ── 4. 모델별 통계 조회 (Grafana용) ──────────────────────────────────────────

@router.get("/stats", response_model=list[StatsResponse])
def get_stats(db: Session = Depends(get_db)):
    """
    모델별 통계를 반환합니다.
    Grafana 대시보드에서 AutoEncoder vs PatchCore 비교에 사용합니다.

    반환 예시:
        [
            { "model_type": "autoencoder", "total": 100, "anomaly_count": 23, ... },
            { "model_type": "patchcore",   "total": 100, "anomaly_count": 18, ... },
        ]
    """
    stats = (
        db.query(
            AnomalyResult.model_type,
            func.count(AnomalyResult.id).label("total_count"),
            func.sum(AnomalyResult.is_anomaly.cast(int)).label("anomaly_count"),
            func.avg(AnomalyResult.anomaly_score).label("avg_score"),
        )
        .group_by(AnomalyResult.model_type)
        .all()
    )

    return [
        StatsResponse(
            model_type   = s.model_type,
            total_count  = s.total_count,
            anomaly_count= s.anomaly_count or 0,
            anomaly_rate = round((s.anomaly_count or 0) / s.total_count, 4),
            avg_score    = round(s.avg_score or 0, 6),
        )
        for s in stats
    ]


# ── 5. 히트맵 이미지 반환 ─────────────────────────────────────────────────────

@router.get("/heatmap/{result_id}")
def get_heatmap(result_id: int, db: Session = Depends(get_db)):
    """
    특정 결과의 히트맵 이미지 파일을 반환합니다.
    대시보드에서 이미지를 직접 보여줄 때 사용합니다.
    """
    result = db.query(AnomalyResult).filter(AnomalyResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail=f"결과를 찾을 수 없습니다: id={result_id}")
    if not result.heatmap_path or not Path(result.heatmap_path).exists():
        raise HTTPException(status_code=404, detail="히트맵 이미지를 찾을 수 없습니다.")

    return FileResponse(result.heatmap_path, media_type="image/png")
