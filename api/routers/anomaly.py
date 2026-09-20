from pathlib import Path
 
import yaml
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session
 
from api.schemas import AnomalyResultList, AnomalyResultResponse, StatsResponse
from db.models import AnomalyResult
from db.session import get_db
 
router = APIRouter(prefix="/anomaly", tags=["anomaly"])
 
with open("config/config.yaml", "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)
 
 
# 모델 선택 / detector 캐시
def get_detector_cls(model_type: str):
    if model_type == "patchcore":
        from models.patchcore.inference import AnomalyDetector
    elif model_type == "autoencoder":
        from models.autoencoder.inference import AnomalyDetector
    else:
        raise ValueError(f"알 수 없는 model.current: {model_type}")
    return AnomalyDetector
 
 
_DETECTOR_CACHE: dict = {}
 
 
def category_from_path(image_path: str, default: str) -> str:
    """data/mvtec/<category>/test/<defect>/000.png 형태에서 카테고리 추출."""
    parts = Path(image_path).parts
    for anchor in ("test", "train"):
        if anchor in parts:
            i = parts.index(anchor)
            if i >= 1:
                return parts[i - 1]
    return default
 
 
def get_detector(category: str):
    """프로세스당 한 번만 로드해 재사용한다."""
    model_type = _config["model"]["current"]
    key = (model_type, category)
 
    if key not in _DETECTOR_CACHE:
        cfg = {**_config, "data": {**_config["data"], "category": category}}
        logger.info(f"Detector 로드: model={model_type} category={category}")
        _DETECTOR_CACHE[key] = get_detector_cls(model_type)(cfg)
 
    return _DETECTOR_CACHE[key]
 
 
# --------------------------------------------------------------------- #
# 엔드포인트
# --------------------------------------------------------------------- #
@router.post("/predict", response_model=AnomalyResultResponse)
def predict(image_path: str, db: Session = Depends(get_db)):
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {image_path}")
 
    category = category_from_path(image_path, _config["data"]["category"])
 
    try:
        detector = get_detector(category)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
 
    result = detector.predict(image_path)
 
    if result["is_anomaly"] is None:
        raise HTTPException(
            status_code=503,
            detail=(f"[{category}] threshold 미설정. "
                    f"python evaluate.py --category {category} 를 먼저 실행하세요."),
        )
 
    db_result = AnomalyResult(
        image_path    = result["image_path"],
        model_type    = result.get("model_type", _config["model"]["current"]),
        anomaly_score = result["score"],
        is_anomaly    = result["is_anomaly"],
        heatmap_path  = result["heatmap_path"],
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
 
 
    logger.info(f"추론 완료 | id={db_result.id} | category={category} "
                f"| score={result['score']:.6f}")
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
            model_type    = s.model_type,
            total_count   = s.total_count,
            anomaly_count = s.anomaly_count or 0,
            anomaly_rate  = round((s.anomaly_count or 0) / s.total_count, 4),
            avg_score     = round(s.avg_score or 0, 6),
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
 