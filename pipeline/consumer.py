"""
pipeline/consumer.py
====================
역할
----
Kafka Topic에서 이미지 경로 메시지를 꺼내서
AutoEncoder로 추론하고 결과를 DB에 저장 + Slack 알림을 보내는 파일입니다.

Consumer란?
-----------
Kafka에서 메시지를 구독(Subscribe)해서 처리하는 역할입니다.
Producer가 Topic에 넣은 메시지를 Consumer가 꺼내서 처리합니다.

Consumer Group이란?
-------------------
config.yaml의 kafka.group_id로 지정합니다.
같은 group_id를 가진 Consumer들은 Topic의 메시지를 나눠서 처리합니다.
Consumer를 여러 개 띄우면 자동으로 부하가 분산됩니다.

처리 흐름
---------
1. Kafka Topic에서 메시지 수신
   → { "image_path": "data/mvtec/bottle/test/good/000.png", "timestamp": ... }
2. 이미지 경로로 파일 읽기
3. AutoEncoder로 추론 → Anomaly Score + 히트맵 생성
4. PostgreSQL에 결과 저장
5. alert_threshold 이상이면 Slack 알림 전송
6. Kafka offset commit (메시지 처리 완료 표시)

offset commit이란?
------------------
Kafka는 메시지를 처리했다는 표시를 offset으로 관리합니다.
처리 완료 후 commit해야 다음 메시지로 넘어갑니다.
enable_auto_commit=False로 설정해서 수동으로 commit합니다.
→ 추론이 실패해도 메시지를 잃지 않고 재처리할 수 있습니다.

실행 방법
---------
  python pipeline/consumer.py
"""

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
    """
    Kafka Consumer 인스턴스를 생성합니다.

    Args:
        topic            : 구독할 Topic 이름
        bootstrap_servers: Kafka 브로커 주소
        group_id         : Consumer Group ID

    Returns:
        KafkaConsumer 인스턴스
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers   = bootstrap_servers,
        group_id            = group_id,
        auto_offset_reset   = "earliest",     # Consumer 처음 시작 시 가장 오래된 메시지부터 처리
        enable_auto_commit  = False,           # 수동 commit (처리 완료 후 직접 commit)
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
    """
    Kafka 메시지 하나를 처리합니다.
    추론 → DB 저장 → Slack 알림 순서로 실행합니다.

    Args:
        message_value: Kafka 메시지 내용 { "image_path": ..., "timestamp": ... }
        detector     : AnomalyDetector 인스턴스
        db           : DB 세션
        config       : 전체 설정 딕셔너리
    """
    image_path = message_value.get("image_path")

    if not image_path or not Path(image_path).exists():
        logger.warning(f"이미지를 찾을 수 없어 스킵: {image_path}")
        return

    # 추론 실행
    result = detector.predict(image_path)

    # DB 저장
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

    # Slack 알림 (alert_threshold 이상일 때만)
    if should_alert(result["score"], config["slack"]["alert_threshold"]):
        send_slack_alert(
            image_path   = result["image_path"],
            score        = result["score"],
            is_anomaly   = result["is_anomaly"],
            heatmap_path = result["heatmap_path"],
            threshold    = config["model"]["threshold"],
        )


def run_consumer(config: dict) -> None:
    """
    Kafka Consumer 메인 루프입니다.
    Topic을 지속적으로 감시하며 메시지가 오면 처리합니다.

    Args:
        config: config.yaml에서 로드한 설정 딕셔너리
    """
    kafka_cfg = config["kafka"]

    # DB 테이블 생성 (없으면 만들고, 있으면 스킵)
    create_tables()

    # 모델 초기화 (1회만)
    detector = AnomalyDetector(config)

    # DB 세션
    db = SessionLocal()

    # Kafka Consumer 생성
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
                # 처리 성공 시 offset commit
                consumer.commit()

            except Exception as e:
                logger.error(f"메시지 처리 실패: {e} | {message.value}")
                db.rollback()
                # 실패해도 offset commit (재처리 루프 방지)
                # 프로덕션에서는 Dead Letter Queue로 보내는 방식 사용
                consumer.commit()

    except KeyboardInterrupt:
        logger.info("Consumer 중단 (KeyboardInterrupt)")

    finally:
        consumer.close()
        db.close()
        logger.info("Consumer 종료")


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_consumer(config)
