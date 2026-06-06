import yaml
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from loguru import logger


def _load_db_url() -> str:
    config_path = Path("config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["database"]["url"]


DATABASE_URL = _load_db_url()

engine = create_engine(
    DATABASE_URL,
    echo=False,          # True로 바꾸면 실행 SQL 터미널에서 확인 가능 (디버깅용)
    pool_pre_ping=True,  # 연결 끊김 시 자동 재연결
)

Base = declarative_base()  # models.py의 AnomalyResult가 이걸 상속

SessionLocal = sessionmaker(
    autocommit=False,  # 명시적 commit() 호출 시에만 저장
    autoflush=False,
    bind=engine,
)


def create_tables() -> None:
    from db import models  # noqa: F401 - Base가 테이블 정보를 인식하도록 import
    Base.metadata.create_all(bind=engine, checkfirst=True)
    logger.info("DB 테이블 생성 완료")


def get_db() -> Generator[Session, None, None]:
    # 요청마다 세션을 열고 요청이 끝나면 (에러가 나도) 반드시 닫음
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()