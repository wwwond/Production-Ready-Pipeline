"""
db/models.py
============
역할
----
PostgreSQL 테이블 구조를 Python 코드로 정의하는 파일입니다.
SQLAlchemy ORM을 사용해서 테이블을 클래스로 표현합니다.

ORM이란?
--------
Object Relational Mapper의 약자입니다.
SQL을 직접 쓰지 않고 Python 클래스로 DB 테이블을 다룰 수 있게 해줍니다.

  SQL 방식    : INSERT INTO anomaly_results (image_path, score) VALUES (...)
  ORM 방식    : db.add(AnomalyResult(image_path=..., score=...))

테이블 구조
-----------
anomaly_results 테이블 하나만 사용합니다.

  id            : 자동 증가 기본키
  image_path    : 추론한 이미지 경로
  model_type    : 사용한 모델 ('autoencoder' or 'patchcore')
                  → AutoEncoder vs PatchCore 비교 쿼리에 핵심 컬럼
  anomaly_score : 이상 점수 (0.0 ~ 1.0)
  is_anomaly    : 이상 여부 (True/False)
  heatmap_path  : 생성된 히트맵 이미지 경로
  created_at    : 추론 시각 (자동 저장)

왜 model_type 컬럼이 중요한가?
------------------------------
나중에 AutoEncoder와 PatchCore 성능 비교할 때 아래 쿼리로 바로 뽑을 수 있어요.

  SELECT model_type, AVG(anomaly_score), COUNT(*)
  FROM anomaly_results
  GROUP BY model_type;

Grafana 대시보드에서도 이 컬럼으로 모델별 필터링이 가능합니다.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from db.session import Base


class AnomalyResult(Base):
    """
    이상 탐지 결과를 저장하는 테이블.

    사용 예시:
        result = AnomalyResult(
            image_path    = "data/mvtec/bottle/test/broken_large/000.png",
            model_type    = "autoencoder",
            anomaly_score = 0.8234,
            is_anomaly    = True,
            heatmap_path  = "results/heatmaps/000_heatmap.png",
        )
        db.add(result)
        db.commit()
    """

    __tablename__ = "anomaly_results"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    image_path    = Column(String,  nullable=False,   comment="추론한 이미지 경로")
    model_type    = Column(String(20), nullable=False, comment="autoencoder or patchcore")
    anomaly_score = Column(Float,   nullable=False,   comment="이상 점수 (MSE 기반)")
    is_anomaly    = Column(Boolean, nullable=False,   comment="이상 여부")
    heatmap_path  = Column(String,  nullable=True,    comment="히트맵 이미지 저장 경로")
    created_at    = Column(
        DateTime(timezone=True),
        server_default=func.now(),  # DB에서 자동으로 현재 시각 저장
        comment="추론 시각"
    )

    def __repr__(self):
        status = "이상" if self.is_anomaly else "정상"
        return (
            f"<AnomalyResult("
            f"id={self.id}, "
            f"model={self.model_type}, "
            f"score={self.anomaly_score:.4f}, "
            f"status={status}"
            f")>"
        )

    def to_dict(self) -> dict:
        """
        FastAPI 응답용 딕셔너리로 변환합니다.
        """
        return {
            "id"           : self.id,
            "image_path"   : self.image_path,
            "model_type"   : self.model_type,
            "anomaly_score": self.anomaly_score,
            "is_anomaly"   : self.is_anomaly,
            "heatmap_path" : self.heatmap_path,
            "created_at"   : str(self.created_at),
        }