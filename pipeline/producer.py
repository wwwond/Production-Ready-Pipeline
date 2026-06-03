"""
pipeline/producer.py
====================
역할
----
MVTec 이미지 폴더를 감시하다가 이미지 파일 경로를 Kafka Topic에 전송하는 파일입니다.

Producer란?
-----------
Kafka에서 메시지를 생성해서 Topic에 보내는 역할입니다.
이 프로젝트에서는 이미지 파일 경로(문자열)만 전송합니다.

왜 이미지 자체를 보내지 않나?
------------------------------
이미지를 Kafka 메시지로 직접 전송하면 용량 문제가 생깁니다.
MVTec 이미지 1장이 약 100KB~1MB인데,
Kafka 메시지 기본 최대 크기는 1MB이고 대용량 처리에 비효율적입니다.

대신 이미지는 공유 볼륨(로컬 폴더)에 저장하고,
Kafka에는 파일 경로(문자열)만 전송합니다.
Consumer가 경로를 받아서 직접 파일을 읽습니다.

  Producer → Kafka: "data/mvtec/bottle/test/good/000.png"  (문자열)
  Consumer ← Kafka: "data/mvtec/bottle/test/good/000.png"  (문자열)
  Consumer → 파일시스템: open("data/mvtec/bottle/test/good/000.png")

프로덕션 전환 시
----------------
Docker Volume 대신 S3/MinIO에 이미지를 업로드하고
Kafka에는 URL을 전송하는 방식으로 전환합니다.
  예: "s3://prp-bucket/images/bottle/000.png"

실행 방법
---------
  python pipeline/producer.py

동작 방식
---------
테스트 이미지 폴더의 이미지를 순서대로 0.5초 간격으로 전송합니다.
실제 제조 라인의 카메라가 이미지를 찍는 상황을 시뮬레이션합니다.
"""

import yaml
import time
import json
from pathlib import Path
from kafka import KafkaProducer
from loguru import logger


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    """
    Kafka Producer 인스턴스를 생성합니다.

    Args:
        bootstrap_servers: Kafka 브로커 주소 (config.yaml의 kafka.bootstrap_servers)

    Returns:
        KafkaProducer 인스턴스
    """
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        # 메시지를 JSON 형식으로 직렬화
        # 경로 외에 메타데이터(타임스탬프 등)도 함께 보낼 수 있도록 dict로 감쌈
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    logger.info(f"Kafka Producer 연결 완료: {bootstrap_servers}")
    return producer


def send_image_path(
    producer: KafkaProducer,
    topic: str,
    image_path: str,
) -> None:
    """
    이미지 파일 경로를 Kafka Topic에 전송합니다.

    전송 메시지 형식:
        {
            "image_path": "data/mvtec/bottle/test/good/000.png",
            "timestamp" : 1717200000.0
        }

    Args:
        producer  : KafkaProducer 인스턴스
        topic     : 전송할 Kafka Topic 이름
        image_path: 전송할 이미지 파일 경로
    """
    message = {
        "image_path": image_path,
        "timestamp" : time.time(),
    }

    # 비동기 전송 후 콜백으로 성공/실패 확인
    future = producer.send(topic, value=message)

    try:
        record = future.get(timeout=10)  # 최대 10초 대기
        logger.info(
            f"전송 완료 | topic={topic} "
            f"partition={record.partition} offset={record.offset} "
            f"| {Path(image_path).name}"
        )
    except Exception as e:
        logger.error(f"전송 실패: {e} | {image_path}")


def run_producer(config: dict) -> None:
    """
    MVTec 테스트 이미지를 순서대로 Kafka에 전송합니다.
    실제 카메라가 이미지를 찍는 상황을 시뮬레이션합니다.

    Args:
        config: config.yaml에서 로드한 설정 딕셔너리
    """
    kafka_cfg = config["kafka"]
    data_cfg  = config["data"]

    producer = create_producer(kafka_cfg["bootstrap_servers"])

    # 테스트 이미지 경로 수집
    test_dir = Path(data_cfg["root"]) / data_cfg["category"] / "test"
    image_paths = sorted(test_dir.rglob("*.png"))

    if not image_paths:
        logger.error(f"이미지를 찾을 수 없습니다: {test_dir}")
        return

    logger.info(f"총 {len(image_paths)}장 전송 시작 | topic={kafka_cfg['topic']}")

    try:
        for image_path in image_paths:
            send_image_path(
                producer   = producer,
                topic      = kafka_cfg["topic"],
                image_path = str(image_path),
            )
            # 0.5초 간격으로 전송 (카메라 촬영 간격 시뮬레이션)
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Producer 중단 (KeyboardInterrupt)")

    finally:
        producer.flush()   # 버퍼에 남은 메시지 모두 전송
        producer.close()
        logger.info("Producer 종료")


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_producer(config)
