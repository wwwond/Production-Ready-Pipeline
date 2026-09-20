import json
import os
from pathlib import Path
 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
 
import yaml
from kafka import KafkaConsumer, KafkaProducer
from loguru import logger
from sqlalchemy.orm import Session
 
from db.models import AnomalyResult
from db.session import SessionLocal, create_tables
 
 
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
 
 
def get_detector(config: dict, category: str):
    """카테고리별 detector 를 캐싱해 재사용한다. memory bank 로드는 비싸다."""
    model_type = config["model"]["current"]
    key = (model_type, category)
 
    if key not in _DETECTOR_CACHE:
        cfg = {**config, "data": {**config["data"], "category": category}}
        logger.info(f"Detector 로드: model={model_type} category={category}")
        _DETECTOR_CACHE[key] = get_detector_cls(model_type)(cfg)
 
    return _DETECTOR_CACHE[key]
 
 
# Kafka
def create_consumer(topic: str, bootstrap_servers: str, group_id: str) -> KafkaConsumer:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers  = bootstrap_servers,
        group_id           = group_id,
        auto_offset_reset  = "earliest",
        enable_auto_commit = False,          # 처리 완료 후에만 수동 commit
        value_deserializer = lambda v: json.loads(v.decode("utf-8")),
    )
    logger.info(f"Kafka Consumer 연결 | topic={topic} group={group_id}")
    return consumer
 
 
def create_dlq_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers = bootstrap_servers,
        value_serializer  = lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
 
 
def send_to_dlq(producer: KafkaProducer, dlq_topic: str, message, error: str) -> None:
    """처리 실패 메시지를 DLQ 로 보낸다. 이후 commit 해도 유실되지 않는다."""
    payload = {
        "original_topic": message.topic,
        "partition"     : message.partition,
        "offset"        : message.offset,
        "value"         : message.value,
        "error"         : error,
    }
    producer.send(dlq_topic, payload)
    producer.flush()
    logger.warning(f"DLQ 전송 | topic={dlq_topic} offset={message.offset} error={error}")
 
 
# 처리
def process_message(message_value: dict, db: Session, config: dict) -> None:
    image_path = message_value.get("image_path")
 
    if not image_path or not Path(image_path).exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
 
    category = category_from_path(image_path, config["data"]["category"])
    detector = get_detector(config, category)
 
    result = detector.predict(image_path)
 
    if result["is_anomaly"] is None:
        raise RuntimeError(
            f"[{category}] threshold 미설정. "
            f"python evaluate.py --category {category} 를 먼저 실행하세요."
        )
 
    db_result = AnomalyResult(
        image_path    = result["image_path"],
        # config 값이 아니라 실제 사용한 모델을 기록한다
        model_type    = result.get("model_type", config["model"]["current"]),
        anomaly_score = result["score"],
        is_anomaly    = result["is_anomaly"],
        heatmap_path  = result["heatmap_path"],
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
 
    logger.info(
        f"DB 저장 | id={db_result.id} | {'이상' if result['is_anomaly'] else '정상'} "
        f"| score={result['score']:.6f} | thr={detector.threshold:.6f}"
    )
 
 
def run_consumer(config: dict) -> None:
    kafka_cfg = config["kafka"]
    dlq_topic = kafka_cfg.get("dlq_topic", f"{kafka_cfg['topic']}-dlq")
 
    create_tables()
 
    db       = SessionLocal()
    consumer = create_consumer(
        topic             = kafka_cfg["topic"],
        bootstrap_servers = kafka_cfg["bootstrap_servers"],
        group_id          = kafka_cfg["group_id"],
    )
    dlq_producer = create_dlq_producer(kafka_cfg["bootstrap_servers"])
 
    logger.info(f"Consumer 대기 중... (model={config['model']['current']}, DLQ={dlq_topic})")
 
    try:
        for message in consumer:
            logger.info(
                f"메시지 수신 | partition={message.partition} offset={message.offset} "
                f"| {message.value.get('image_path', '')}"
            )
            try:
                process_message(message.value, db, config)
                consumer.commit()
 
            except Exception as e:
                logger.error(f"처리 실패: {e} | {message.value}")
                db.rollback()
                # DLQ 로 보낸 뒤에만 commit 한다. 전송 실패 시 commit 하지 않아
                # 메시지가 보존되고 재시작 시 재처리된다.
                try:
                    send_to_dlq(dlq_producer, dlq_topic, message, str(e))
                    consumer.commit()
                except Exception as dlq_err:
                    logger.error(f"DLQ 전송 실패 -> commit 보류: {dlq_err}")
 
    except KeyboardInterrupt:
        logger.info("Consumer 중단 (KeyboardInterrupt)")
 
    finally:
        consumer.close()
        dlq_producer.close()
        db.close()
        logger.info("Consumer 종료")
 
 
if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
 
    run_consumer(config)
 