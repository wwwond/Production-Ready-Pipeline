from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AnomalyResultResponse(BaseModel):
    id            : int
    image_path    : str
    model_type    : str
    anomaly_score : float
    is_anomaly    : bool
    heatmap_path  : Optional[str] = None
    created_at    : datetime

    class Config:
        from_attributes = True


class AnomalyResultList(BaseModel):
    total  : int
    results: list[AnomalyResultResponse]


class StatsResponse(BaseModel):
    model_type      : str
    total_count     : int
    anomaly_count   : int
    anomaly_rate    : float   
    avg_score       : float