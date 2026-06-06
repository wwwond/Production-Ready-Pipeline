# PRP - Precision Realtime Pipeline

> 제조 공정 이상 탐지 시스템 + 실시간 알림 파이프라인

---

## 프로젝트 개요

MVTec AD 데이터셋을 활용한 제조 공정 이상 탐지 시스템입니다.
CNN AutoEncoder와 PatchCore 두 모델을 직접 구현하고 성능을 비교했습니다.
탐지 결과는 Kafka 파이프라인을 통해 실시간으로 처리되며 FastAPI, PostgreSQL, Grafana로 서빙됩니다.

---

---

## Notion 기록
[Notion Link] -> (https://accurate-seed-914.notion.site/PRP-372f17ee028580628350ceb1bf923c2c?source=copy_link)

---

## 실험 결과

### AutoEncoder vs PatchCore (carpet 카테고리)

| 모델 | 정상 score 평균 | 이상 score 평균 | 이상 탐지율 | 오탐율 |
|------|----------------|----------------|-------------|--------|
| CNN AutoEncoder | 0.001189 | 0.001058 | 0% | - |
| PatchCore | 3.467 | 4.838 | **75.3%** (67/89장) | 3.6% (1/28장) |

### 분석

AutoEncoder는 정상/이상 score가 역전되어 탐지 실패했습니다.
carpet처럼 복잡한 텍스처는 이상 이미지도 비슷하게 복원되어 복원 오차 차이가 거의 없었습니다.

PatchCore는 사전학습된 ResNet으로 패치 단위 특징을 추출하고
Memory Bank와의 거리로 판단하여 score 분리에 성공했습니다.

> "AutoEncoder의 한계를 확인하고 PatchCore로 교체하여 0% → 75.3% 탐지율을 달성했습니다."

---

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

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 데이터 | MVTec AD |
| 모델 | CNN AutoEncoder → PatchCore (Wide ResNet50) |
| 파이프라인 | Kafka |
| 시각화 | OpenCV (Anomaly Map) |
| 서빙 | FastAPI |
| 저장 | PostgreSQL |
| 대시보드 | Grafana |
| 알림 | Slack Webhook |
| 배포 | Docker Compose |

---

## 프로젝트 구조

```
prp/
├── api/                        # FastAPI 앱
│   ├── routers/anomaly.py      # 추론/조회/통계/히트맵 엔드포인트
│   ├── schemas.py              # 요청/응답 데이터 형식
│   └── main.py                 # 앱 진입점
├── config/
│   └── config.yaml             # 전체 설정값 (모델, Kafka, DB 등)
├── data/
│   └── mvtec/                  # MVTec AD 데이터셋
├── db/
│   ├── models.py               # anomaly_results 테이블 정의
│   └── session.py              # DB 연결 및 세션 관리
├── docker/                     # Dockerfile 모음
├── models/
│   ├── autoencoder/            # 1단계: CNN AutoEncoder
│   │   ├── model.py            # Encoder + Decoder 구조
│   │   ├── train.py            # 정상 이미지만으로 학습
│   │   └── inference.py        # 추론 + Anomaly Score + 히트맵
│   └── patchcore/              # 2단계: PatchCore
│       ├── model.py            # ResNet 특징 추출 + Memory Bank
│       ├── train.py            # Memory Bank 구축
│       └── inference.py        # Memory Bank 거리 기반 추론
├── pipeline/
│   ├── producer.py             # 이미지 경로를 Kafka에 전송
│   └── consumer.py             # 추론 + DB 저장 + Slack 알림
├── utils/
│   ├── visualize.py            # 히트맵 생성 (원본/복원/히트맵 비교)
│   └── notify.py               # Slack Webhook 알림
├── .env.example                # 환경변수 예시
├── docker-compose.yml          # 전체 서비스 구성
└── requirements.txt
```

---

## 시작하기

### 1. 환경 설정

```bash
git clone https://github.com/wwwond/Production-Ready-Pipeline.git
cd Production-Ready-Pipeline
cp .env.example .env      # .env 파일 생성 후 값 채우기
```

### 2. 데이터 다운로드

[MVTec AD 공식 페이지](https://www.mvtec.com/company/research/datasets/mvtec-ad)에서 다운로드 후 `data/mvtec/`에 위치시킵니다.

```
data/mvtec/
├── bottle/
├── carpet/     ← 실험에 사용
├── leather/
└── ...
```

### 3. 로컬 실행 (모델만)

```bash
# PatchCore Memory Bank 구축
python models/patchcore/train.py

# 추론 실행
python models/patchcore/inference.py
```

### 4. 전체 파이프라인 실행

```bash
docker compose up -d
docker compose run producer    # 이미지 전송 시작
```

### 5. 결과 확인

| 서비스 | 주소 |
|--------|------|
| FastAPI Swagger | http://localhost:8000/docs |
| Grafana 대시보드 | http://localhost:3000 (admin/admin) |

---

## 주요 설계 결정

**왜 Kafka에 이미지를 직접 넣지 않았나?**
이미지를 직접 전송하면 Kafka 메시지 크기 제한(기본 1MB)에 걸리고 처리 속도가 느려집니다.
대신 이미지는 공유 볼륨에 저장하고 Kafka에는 파일 경로(문자열)만 전송합니다.
프로덕션 환경에서는 이미지를 S3/MinIO에 업로드하고 URL을 전송하는 방식으로 전환합니다.

**왜 Kafka offset을 수동으로 commit하나?**
`enable_auto_commit=False`로 설정해서 추론이 성공한 후에만 commit합니다.
추론 도중 에러가 나도 메시지를 잃지 않고 재처리할 수 있습니다.

**왜 model_type 컬럼을 별도로 두나?**
AutoEncoder와 PatchCore 결과를 같은 테이블에 저장하고
`model_type` 컬럼으로 필터링해서 성능을 직접 비교할 수 있습니다.

---

## 개발 단계

- [x] 아키텍처 설계
- [x] CNN AutoEncoder 구현 + 실험 (한계 확인)
- [x] PatchCore 구현 + 성능 비교 (75.3% 달성)
- [x] Kafka 파이프라인 연결
- [x] FastAPI + PostgreSQL 서빙
- [x] Docker Compose 전체 동작 확인
- [ ] Grafana 대시보드 구성
- [ ] Slack Webhook 연동
- [ ] 전체 카테고리 실험 및 결과 정리