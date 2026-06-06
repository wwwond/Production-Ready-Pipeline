import yaml
import json
from pathlib import Path
from kafka import KafkaConsumer
from loguru import logger
from sqlalchemy.orm import Session

from db.session import SessionLocal, create_tables
from db.models import AnomalyResult
from models.autoencoder.inference import AnomalyDetector
from utils.notify import send_slack_alert, should_alert


def create_consumer(
    topic: str,
    bootstrap_servers: str,
    group_id: str,
) -> KafkaConsumer:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers   = bootstrap_servers,
        group_id            = group_id,
        auto_offset_reset   = "earliest",     
        enable_auto_commit  = False,           
        value_deserializer  = lambda v: json.loads(v.decode("utf-8")),
    )
    logger.info(f"Kafka Consumer 연결 완료 | topic={topic} group={group_id}")
    return consumer


def process_message(
    message_value: dict,
    detector: AnomalyDetector,
    db: Session,
    config: dict,
) -> None:
    image_path = message_value.get("image_path")

    if not image_path or not Path(image_path).exists():
        logger.warning(f"이미지를 찾을 수 없어 스킵: {image_path}")
        return

    result = detector.predict(image_path)

    db_result = AnomalyResult(
        image_path    = result["image_path"],
        model_type    = config["model"]["current"],
        anomaly_score = result["score"],
        is_anomaly    = result["is_anomaly"],
        heatmap_path  = result["heatmap_path"],
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    logger.info(
        f"DB 저장 완료 | id={db_result.id} "
        f"| {'🔴 이상' if result['is_anomaly'] else '🟢 정상'} "
        f"| score={result['score']:.6f}"
    )

    if should_alert(result["score"], config["slack"]["alert_threshold"]):
        send_slack_alert(
            image_path   = result["image_path"],
            score        = result["score"],
            is_anomaly   = result["is_anomaly"],
            heatmap_path = result["heatmap_path"],
            threshold    = config["model"]["threshold"],
        )


def run_consumer(config: dict) -> None:
    kafka_cfg = config["kafka"]

    create_tables()

    detector = AnomalyDetector(config)

    db = SessionLocal()

    consumer = create_consumer(
        topic             = kafka_cfg["topic"],
        bootstrap_servers = kafka_cfg["bootstrap_servers"],
        group_id          = kafka_cfg["group_id"],
    )

    logger.info("Consumer 대기 중... (메시지를 기다립니다)")

    try:
        for message in consumer:
            logger.info(
                f"메시지 수신 | partition={message.partition} "
                f"offset={message.offset} | {message.value.get('image_path', '')}"
            )

            try:
                process_message(
                    message_value = message.value,
                    detector      = detector,
                    db            = db,
                    config        = config,
                )
                consumer.commit()

            except Exception as e:
                logger.error(f"메시지 처리 실패: {e} | {message.value}")
                db.rollback()
                consumer.commit()

    except KeyboardInterrupt:
        logger.info("Consumer 중단 (KeyboardInterrupt)")

    finally:
        consumer.close()
        db.close()
        logger.info("Consumer 종료")


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_consumer(config)