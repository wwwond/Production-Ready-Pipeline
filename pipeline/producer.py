import yaml
import time
import json
from pathlib import Path
from kafka import KafkaProducer
from loguru import logger


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    logger.info(f"Kafka Producer 연결 완료: {bootstrap_servers}")
    return producer


def send_image_path(
    producer: KafkaProducer,
    topic: str,
    image_path: str,
) -> None:
    message = {
        "image_path": image_path,
        "timestamp" : time.time(),
    }

    future = producer.send(topic, value=message)

    try:
        record = future.get(timeout=10)  
        logger.info(
            f"전송 완료 | topic={topic} "
            f"partition={record.partition} offset={record.offset} "
            f"| {Path(image_path).name}"
        )
    except Exception as e:
        logger.error(f"전송 실패: {e} | {image_path}")


def run_producer(config: dict) -> None:
    kafka_cfg = config["kafka"]
    data_cfg  = config["data"]

    producer = create_producer(kafka_cfg["bootstrap_servers"])

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
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Producer 중단 (KeyboardInterrupt)")

    finally:
        producer.flush()   
        producer.close()
        logger.info("Producer 종료")


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_producer(config)