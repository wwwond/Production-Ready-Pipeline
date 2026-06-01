"""
db/session.py
=============
역할
----
PostgreSQL 데이터베이스 연결을 관리하는 파일입니다.
SQLAlchemy 엔진과 세션을 생성하고, FastAPI에서 DB를 사용할 수 있게 해줍니다.

구성 요소
---------
1. engine      : DB 연결 엔진 (실제 PostgreSQL에 접속)
2. Base        : 모든 테이블 클래스의 부모 클래스 (db/models.py에서 상속)
3. SessionLocal: DB 작업을 위한 세션 팩토리
4. get_db()    : FastAPI 의존성 주입용 세션 제공 함수

왜 get_db()를 따로 만드나?
--------------------------
FastAPI에서 DB 세션을 안전하게 열고 닫기 위해서입니다.
요청이 들어오면 세션을 열고, 요청이 끝나면 (에러가 나도) 세션을 닫습니다.

  @app.get("/results")
  def get_results(db: Session = Depends(get_db)):
      return db.query(AnomalyResult).all()

연결 설정
---------
config.yaml의 database.url 값을 사용합니다.
  postgresql://prp_user:prp_pass@postgres:5432/prp_db
  └─ 드라이버  └─ 유저    └─ 비밀번호  └─ 호스트  └─ 포트  └─ DB명

로컬 개발 vs Docker 환경
------------------------
- 로컬  : postgresql://prp_user:prp_pass@localhost:5432/prp_db
- Docker: postgresql://prp_user:prp_pass@postgres:5432/prp_db
          (호스트가 localhost → postgres 컨테이너명으로 변경)
"""

import yaml
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from loguru import logger

# ── 설정 로드 ─────────────────────────────────────────────────────────────────

def _load_db_url() -> str:
    """
    config.yaml에서 DB 연결 URL을 읽어옵니다.
    """
    config_path = Path("config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["database"]["url"]


DATABASE_URL = _load_db_url()

# ── SQLAlchemy 설정 ───────────────────────────────────────────────────────────

# 엔진: 실제 PostgreSQL에 연결하는 객체
engine = create_engine(
    DATABASE_URL,
    echo=False,       # True로 바꾸면 실행되는 SQL을 터미널에서 볼 수 있음 (디버깅용)
    pool_pre_ping=True,  # 연결이 끊겼을 때 자동으로 재연결
)

# Base: 모든 테이블 클래스가 상속받는 부모 클래스
# db/models.py의 AnomalyResult가 이걸 상속받음
Base = declarative_base()

# SessionLocal: DB 세션을 만드는 팩토리
SessionLocal = sessionmaker(
    autocommit=False,  # 명시적으로 commit() 호출해야 저장됨
    autoflush=False,
    bind=engine,
)

# ── 테이블 생성 ───────────────────────────────────────────────────────────────

def create_tables() -> None:
    """
    정의된 모든 테이블을 DB에 생성합니다.
    이미 존재하는 테이블은 건드리지 않습니다.

    FastAPI 앱 시작 시 한 번 호출합니다 (api/main.py).
    """
    # models.py를 import해야 Base가 테이블 정보를 알 수 있음
    from db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("DB 테이블 생성 완료")


# ── FastAPI 의존성 주입용 세션 ────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI 라우터에서 DB 세션을 안전하게 사용하기 위한 함수입니다.
    요청마다 세션을 열고, 요청이 끝나면 자동으로 닫습니다.

    사용 예시 (api/routers/anomaly.py):
        @router.get("/results")
        def get_results(db: Session = Depends(get_db)):
            return db.query(AnomalyResult).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
