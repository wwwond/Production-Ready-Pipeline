"""
api/schemas.py
==============
역할
----
FastAPI에서 요청(Request)과 응답(Response)의 데이터 형식을 정의하는 파일입니다.
Pydantic 모델을 사용해서 데이터 유효성 검사와 직렬화를 처리합니다.

Pydantic이란?
-------------
Python 타입 힌트를 기반으로 데이터 유효성을 자동 검사해주는 라이브러리입니다.
FastAPI가 내부적으로 Pydantic을 사용해서 요청/응답 데이터를 검증합니다.

왜 schemas.py를 별도로 분리하나?
---------------------------------
DB 모델(db/models.py)과 API 응답 형식을 분리하기 위해서입니다.
DB에는 민감한 내부 정보가 있을 수 있고,
API 응답은 클라이언트에게 필요한 데이터만 골라서 보내야 합니다.

  DB 모델    → SQLAlchemy  → DB 저장/조회용
  API 스키마 → Pydantic    → 요청/응답 형식 검증용

스키마 구성
-----------
1. AnomalyResultResponse : 추론 결과 응답 형식
2. AnomalyResultList     : 결과 목록 응답 형식
3. StatsResponse         : 통계 응답 형식 (Grafana 대시보드용)
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AnomalyResultResponse(BaseModel):
    """
    추론 결과 단건 응답 스키마.

    GET /results/{id}
    POST /predict 응답에 사용됩니다.
    """
    id            : int
    image_path    : str
    model_type    : str
    anomaly_score : float
    is_anomaly    : bool
    heatmap_path  : Optional[str] = None
    created_at    : datetime

    class Config:
        # SQLAlchemy 모델을 Pydantic 모델로 자동 변환 허용
        from_attributes = True


class AnomalyResultList(BaseModel):
    """
    추론 결과 목록 응답 스키마.

    GET /results 응답에 사용됩니다.
    """
    total  : int
    results: list[AnomalyResultResponse]


class StatsResponse(BaseModel):
    """
    모델별 통계 응답 스키마.
    Grafana 대시보드 및 GET /stats 엔드포인트에 사용됩니다.

    model_type별로 아래 통계를 반환합니다.
      - 전체 추론 수
      - 이상 탐지 수
      - 이상 탐지율
      - 평균 Anomaly Score
    """
    model_type      : str
    total_count     : int
    anomaly_count   : int
    anomaly_rate    : float   # 이상 탐지율 (0.0 ~ 1.0)
    avg_score       : float   # 평균 Anomaly Score
