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

with open("config/config.yaml", "r") as f:
    _config = yaml.safe_load(f)


def _get_detector() -> AnomalyDetector:
    return AnomalyDetector(_config)


@router.post("/predict", response_model=AnomalyResultResponse)
def predict(image_path: str, db: Session = Depends(get_db)):
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {image_path}")

    detector = _get_detector()
    result   = detector.predict(image_path)

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


@router.get("/results", response_model=AnomalyResultList)
def get_results(
    model_type: str = None,   
    is_anomaly: bool = None,  
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(AnomalyResult)

    if model_type:
        query = query.filter(AnomalyResult.model_type == model_type)
    if is_anomaly is not None:
        query = query.filter(AnomalyResult.is_anomaly == is_anomaly)

    total   = query.count()
    results = query.order_by(AnomalyResult.created_at.desc()).offset(offset).limit(limit).all()

    return AnomalyResultList(total=total, results=results)


@router.get("/results/{result_id}", response_model=AnomalyResultResponse)
def get_result(result_id: int, db: Session = Depends(get_db)):
    result = db.query(AnomalyResult).filter(AnomalyResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail=f"결과를 찾을 수 없습니다: id={result_id}")
    return result


@router.get("/stats", response_model=list[StatsResponse])
def get_stats(db: Session = Depends(get_db)):
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


@router.get("/heatmap/{result_id}")
def get_heatmap(result_id: int, db: Session = Depends(get_db)):
    result = db.query(AnomalyResult).filter(AnomalyResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail=f"결과를 찾을 수 없습니다: id={result_id}")
    if not result.heatmap_path or not Path(result.heatmap_path).exists():
        raise HTTPException(status_code=404, detail="히트맵 이미지를 찾을 수 없습니다.")

    return FileResponse(result.heatmap_path, media_type="image/png")