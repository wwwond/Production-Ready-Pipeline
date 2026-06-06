from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from db.session import Base


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"

    id            = Column(Integer,     primary_key=True, autoincrement=True)
    image_path    = Column(String,      nullable=False)
    model_type    = Column(String(30),  nullable=False)   # "autoencoder" or "patchcore"
    anomaly_score = Column(Float,       nullable=False)
    is_anomaly    = Column(Boolean,     nullable=False)
    heatmap_path  = Column(String,      nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())  # DB 자동 저장

    def __repr__(self):
        status = "이상" if self.is_anomaly else "정상"
        return f"<AnomalyResult(id={self.id}, model={self.model_type}, score={self.anomaly_score:.4f}, status={status})>"

    def to_dict(self) -> dict:
        return {
            "id"           : self.id,
            "image_path"   : self.image_path,
            "model_type"   : self.model_type,
            "anomaly_score": self.anomaly_score,
            "is_anomaly"   : self.is_anomaly,
            "heatmap_path" : self.heatmap_path,
            "created_at"   : str(self.created_at),
        }