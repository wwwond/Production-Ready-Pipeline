# PRP - Precision Realtime Pipeline

> 제조 공정 이상 탐지 시스템 + 실시간 알림 파이프라인

## 프로젝트 개요

MVTec AD 데이터셋을 활용한 제조 공정 이상 탐지 시스템입니다.  
CNN AutoEncoder → PatchCore 순서로 모델을 개선하며 성능을 비교합니다.

## 아키텍처

```
[이미지 폴더 (MVTec AD)]
        ↓
[Kafka Producer: 파일 경로만 전송]
        ↓
[Kafka Topic: anomaly-detection]
        ↓
[Kafka Consumer: 경로 받아서 이미지 로드]
        ↓
[1단계: CNN AutoEncoder 추론]
[2단계: PatchCore 추론으로 교체 + 성능 비교]
        ↓
[Anomaly Map 생성 (OpenCV 히트맵)]
        ↓
[PostgreSQL 저장]
        ↓
[FastAPI] ── [Slack Webhook 알림]
        ↓
[Grafana 대시보드]
```

## 기술 스택

| 분류 | 기술 |
|------|------|
| 데이터 | MVTec AD |
| 파이프라인 | Kafka |
| 모델 | CNN AutoEncoder → PatchCore |
| 시각화 | OpenCV (Anomaly Map) |
| 서빙 | FastAPI |
| 저장 | PostgreSQL |
| 대시보드 | Grafana |
| 알림 | Slack Webhook |
| 배포 | Docker Compose |

## 프로젝트 구조

```
prp/
├── api/                  # FastAPI 앱
│   ├── routers/
│   │   └── anomaly.py
│   ├── schemas.py
│   └── main.py
├── config/
│   └── config.yaml       # 전체 설정값
├── data/
│   └── mvtec/            # MVTec AD 데이터셋
├── db/
│   ├── models.py         # SQLAlchemy 테이블 정의
│   └── session.py        # DB 연결
├── docker/               # Dockerfile 모음
├── models/
│   ├── autoencoder/      # 1단계 모델
│   └── patchcore/        # 2단계 모델
├── pipeline/
│   ├── producer.py       # Kafka Producer
│   └── consumer.py       # Kafka Consumer
├── results/
│   ├── checkpoints/      # 모델 가중치
│   ├── heatmaps/         # Anomaly Map 이미지
│   └── metrics/          # 실험 결과
├── utils/
│   ├── visualize.py      # 히트맵 생성
│   └── notify.py         # Slack Webhook
├── .env.example
├── docker-compose.yml
└── requirements.txt
```

## 시작하기

### 1. 환경 설정

```bash
git clone https://github.com/your-id/prp.git
cd prp
cp .env.example .env      # .env 파일 생성 후 값 채우기
pip install -r requirements.txt
```

### 2. 데이터 다운로드

[MVTec AD 공식 페이지](https://www.mvtec.com/company/research/datasets/mvtec-ad)에서 다운로드 후 `data/mvtec/`에 위치시킵니다.

### 3. 로컬 실행 (1단계 - 모델만)

```bash
python models/autoencoder/train.py
python models/autoencoder/inference.py
```

### 4. 전체 파이프라인 실행

```bash
docker compose up -d
```

## 개발 단계

- [x] 아키텍처 설계
- [x] 환경 설정 (config.yaml, requirements.txt)
- [ ] CNN AutoEncoder 구현 + 로컬 동작 확인
- [ ] FastAPI + PostgreSQL 연결
- [ ] Kafka 파이프라인 연결
- [ ] Grafana 대시보드
- [ ] PatchCore 교체 + 성능 비교

## 프로덕션 전환 시 고려사항

현재는 Docker Volume 마운트로 Producer/Consumer 간 파일 경로를 공유합니다.  
실제 제조 환경에서는 이미지를 **S3 또는 MinIO**에 업로드하고, Kafka에는 URL만 전송하는 방식으로 전환합니다.