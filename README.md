# PRP - Precision Realtime Pipeline

> 제조 공정 이상 탐지 시스템 + 실시간 처리 파이프라인

---

## 프로젝트 개요

MVTec AD 데이터셋을 활용한 제조 공정 이상 탐지 시스템입니다.
CNN AutoEncoder와 PatchCore 두 모델을 직접 구현해 비교하고, MVTec AD 15개 카테고리
전체에 대해 정량 평가했습니다. 탐지 결과는 Kafka 파이프라인을 통해 실시간으로
처리되어 FastAPI와 PostgreSQL로 서빙됩니다.

모든 실험은 외장 GPU 없이 CPU(Intel Core Ultra 7 258V)에서 수행했습니다.

---

## Notion 기록
[Notion Link](https://accurate-seed-914.notion.site/PRP-372f17ee028580628350ceb1bf923c2c?source=copy_link)

---

## 실험 결과

### MVTec AD 전체 15개 카테고리 (PatchCore, greedy coreset 1%)

| category | image AUROC | pixel AUROC | recall@thr | FPR@thr | recall@FPR=0 |
|---|---|---|---|---|---|
| bottle | 1.000 | 0.986 | 1.000 | 0.000 | 1.000 |
| cable | 0.994 | 0.986 | 0.978 | 0.103 | 0.880 |
| capsule | 0.990 | 0.989 | 0.789 | 0.000 | 0.954 |
| carpet | 0.989 | 0.990 | 0.989 | 0.286 | 0.933 |
| grid | 0.974 | 0.975 | 0.860 | 0.048 | 0.842 |
| hazelnut | 1.000 | 0.987 | 0.986 | 0.000 | 1.000 |
| leather | 1.000 | 0.993 | 1.000 | 0.094 | 1.000 |
| metal_nut | 0.997 | 0.984 | 0.957 | 0.000 | 0.957 |
| pill | 0.955 | 0.974 | 0.780 | 0.038 | 0.745 |
| screw | 0.927 | 0.976 | 0.697 | 0.098 | 0.294 |
| tile | 1.000 | 0.958 | 0.917 | 0.000 | 1.000 |
| toothbrush | 0.922 | 0.989 | 0.967 | 0.250 | 0.633 |
| transistor | 0.996 | 0.960 | 0.975 | 0.017 | 0.925 |
| wood | 0.986 | 0.942 | 0.967 | 0.158 | 0.917 |
| zipper | 0.975 | 0.986 | 0.992 | 0.094 | 0.546 |
| **평균** | **0.980** | **0.978** | | | **0.842** |

원 논문(PatchCore-1%) 평균 image AUROC 0.990 / pixel AUROC 0.981 대비,
GPU 없이 CPU만으로 image 1.0%p 차이, pixel 동등 수준으로 재현했습니다.

**두 임계값의 성격이 다릅니다.**

| 지표 | 산출 기준 | 용도 |
|---|---|---|
| `threshold` | 학습용 정상 이미지 holdout(20%)의 p99 | 테스트셋을 보지 않는 **운영 임계값** |
| `recall@FPR=0` | 테스트 정상 score 최댓값 | ROC 곡선의 FPR=0 지점을 읽는 **진단 지표**. 운영에는 사용 불가 |

### 고도화 전후 (carpet)

| 항목 | 이전 | 현재 |
|---|---|---|
| recall @ FPR=0 | 75.3% | **93.3%** |
| image AUROC | 미측정 | **0.989** |
| pixel AUROC | 미측정 | **0.990** |
| memory bank | 176 MB (10% random) | **13.4 MB** (1% greedy) |
| 재현성 | 실행마다 ±5%p 변동 | 시드 고정, 결정론적 |
| 임계값 산출 | 테스트셋 정상 최댓값 (누수) | 학습 holdout p99 |

주요 변경 사항:

1. **local neighborhood aggregation(3x3 avg pool) 추가** — 논문 3.1의 locally aware
   patch feature. 각 위치를 3x3 이웃 평균으로 대체해 수용영역을 넓히고 패치 표현의
   노이즈 민감도를 낮춥니다.
2. **random subsampling → greedy k-center coreset** — 논문의 핵심 기여를 실제로 구현.
   기존 코드는 SparseRandomProjection의 반환값을 버리고 균등 랜덤 샘플링을 하고 있어,
   논문의 coreset이 아니었을 뿐 아니라 시드가 없어 실행마다 결과가 달라졌습니다.
   거리 계산만 128차원 축소 공간에서 수행하고(Johnson-Lindenstrauss) 선택된 인덱스로
   원본 차원의 특징을 반환합니다.
3. **ball_tree → brute force NN 검색** — 특징 차원이 1536이라 트리 기반 인덱스는
   가지치기가 동작하지 않고 순회 오버헤드만 추가됩니다. torch.cdist로 BLAS 행렬곱을
   타도록 변경했습니다.

### AutoEncoder vs PatchCore (carpet)

| 모델 | 집계 | image AUROC | pixel AUROC |
|---|---|---|---|
| CNN AutoEncoder | mean | 0.341 | 0.691 |
| CNN AutoEncoder | max | 0.323 | 0.691 |
| PatchCore | max | **0.989** | **0.990** |

---

## 분석

### AutoEncoder의 실패는 성능 부족이 아니라 신호의 역전

image AUROC 0.341은 무작위(0.5)보다 낮습니다. 방향을 뒤집으면 0.659가 되므로
"복원 오차가 작을수록 이상"이라는 역관계가 실제로 존재합니다.

원인은 carpet의 데이터 특성입니다. 결함(색 얼룩, 구멍, 실밥)은 주변 카펫 텍스처보다
평탄해서 AE가 오히려 쉽게 복원합니다. 반면 복잡한 텍스처 자체가 가장 큰 복원 오차를
만듭니다. 실제로 이상 이미지의 평균 score가 정상보다 낮습니다 (0.001058 < 0.001189).

집계 방식을 mean에서 max로 바꿔도 0.341 → 0.323으로 변화가 없습니다. PatchCore가
max를 쓰므로 비교 조건을 맞춘 통제 실험인데, 결과가 같다는 것은 실패 원인이 집계가
아니라 복원 오차 지표 자체에 있음을 보여줍니다.

구조적 배경으로, latent가 512x16x16 = 131,072차원이어서 입력(3x256x256 = 196,608차원)
대비 압축률이 1.5배에 불과합니다. "좁은 병목은 이상을 복원하지 못한다"는 AE 이상탐지의
전제가 성립하지 않아, 이상 영역까지 그대로 복원됩니다.

### AUROC가 높아도 운영점 성능은 보장되지 않는다

zipper는 image AUROC 0.975로 양호하지만 오탐 0 조건에서는 54.6%만 탐지합니다.
정상 이미지 중 소수가 극단적으로 높은 score를 가져(최댓값 2.80, 정상 평균 1.72)
운영 임계값을 끌어올리기 때문입니다. screw(AUROC 0.927 / recall@FPR=0 0.294)도
같은 양상입니다.

AUROC는 ROC 곡선 전체를 요약하는 지표이므로, 실제 배포에 쓰이는 특정 운영점의
성능과 괴리가 발생할 수 있습니다. 모델 선택 시 두 지표를 함께 보아야 합니다.

### 학습 시점 임계값이 운영 데이터에서 유효하지 않다

carpet에서 학습 데이터 기준으로 산출한 threshold의 FPR이 테스트에서 28.6%로
상승했습니다. train/good과 test/good의 score 분포가 동일하지 않다는 뜻입니다.

운영 중 정상 score 분포를 모니터링하고 임계값을 주기적으로 재보정해야 합니다.

### screw / toothbrush가 낮은 이유

- **screw (0.927)**: 대상이 임의 각도로 회전 촬영되어, memory bank에 없는 각도가
  정상임에도 큰 거리를 갖습니다. 결함(미세 긁힘)이 회전 변동보다 작아 신호가 묻힙니다.
  논문에서도 PatchCore의 최약 카테고리입니다.
- **toothbrush (0.922)**: train 이미지가 60장뿐이라 coreset 1% = 491개로 memory bank가
  다른 카테고리(1,700~3,200개)의 1/6 수준입니다. 또한 정상 테스트가 12장이라
  AUROC 자체의 분산이 큽니다.

### 지표 해석 주의

pixel AUROC는 결함 픽셀이 전체의 1% 미만인 극심한 클래스 불균형 때문에 낙관적으로
측정됩니다. 보다 엄밀한 비교에는 AUPRO가 적절합니다.

---

## 실시간 파이프라인 검증

Kafka → PatchCore 추론 → PostgreSQL 저장 전 구간을 Docker Compose 환경에서
동작 확인했습니다.

- **처리 시간**: 메시지 수신 → DB 저장 약 280ms/건 (CPU). 모델은 (모델, 카테고리)별로
  프로세스당 1회만 로드하고 캐싱합니다.
- **실패 처리**: 존재하지 않는 이미지 경로를 주입해 DLQ 동작을 검증했습니다.
  원본 토픽/파티션/오프셋/메시지/에러 사유를 보존한 채 `anomaly-detection-dlq`로
  이관된 뒤 commit되어, 메시지가 유실되지 않으면서 poison message로 인한
  무한 재처리도 발생하지 않습니다.

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
[PatchCore 추론 (AutoEncoder와 동일 인터페이스로 교체 가능)]
        ↓
[Anomaly Map 생성 (OpenCV 히트맵)]
        ↓
[PostgreSQL 저장] ──→ [FastAPI 조회/통계 API]
        │
        └─ 처리 실패 시 ──→ [Kafka DLQ: anomaly-detection-dlq]
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 데이터 | MVTec AD |
| 모델 | CNN AutoEncoder → PatchCore (Wide ResNet50-2) |
| 평가 | scikit-learn (AUROC, AUPR), SciPy |
| 파이프라인 | Kafka (수동 commit + DLQ) |
| 시각화 | OpenCV (Anomaly Map) |
| 서빙 | FastAPI |
| 저장 | PostgreSQL |
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
│   ├── autoencoder/            # 베이스라인: CNN AutoEncoder
│   │   ├── model.py            # Encoder + Decoder 구조
│   │   ├── train.py            # 정상 이미지만으로 학습
│   │   └── inference.py        # 복원 오차 기반 추론
│   └── patchcore/              # 본 모델: PatchCore
│       ├── model.py            # 특징 추출 + greedy coreset + Memory Bank
│       ├── train.py            # Memory Bank 구축 + holdout 분리
│       └── inference.py        # Memory Bank 거리 기반 추론
├── pipeline/
│   ├── producer.py             # 이미지 경로를 Kafka에 전송
│   └── consumer.py             # 추론 + DB 저장 + DLQ 처리
├── utils/
│   └── visualize.py            # 히트맵 생성
├── evaluate.py                 # 정량 평가 (AUROC, 임계값 산출)
├── .env.example
├── docker-compose.yml
└── requirements.txt
```

---

## 시작하기

### 1. 환경 설정

```bash
git clone https://github.com/wwwond/Production-Ready-Pipeline.git
cd Production-Ready-Pipeline
cp .env.example .env
```

### 2. 데이터 다운로드

[MVTec AD 공식 페이지](https://www.mvtec.com/company/research/datasets/mvtec-ad)에서
다운로드 후 `data/mvtec/`에 위치시킵니다.

```
data/mvtec/
├── bottle/
├── carpet/
├── leather/
└── ...
```

### 3. Memory Bank 구축 및 평가

```bash
# 특정 카테고리
python models/patchcore/train.py --category carpet
python evaluate.py --category carpet

# 전체 15개 카테고리
for c in bottle cable capsule carpet grid hazelnut leather metal_nut \
         pill screw tile toothbrush transistor wood zipper; do
  python models/patchcore/train.py --category $c
done
python evaluate.py --all

# 베이스라인 AutoEncoder와 비교
python evaluate.py --model autoencoder --category carpet
```

`evaluate.py`가 `results/metrics/{category}.json`에 지표와 운영 임계값을 기록하고,
추론 시 이 값을 읽습니다. 따라서 **train → evaluate 순서로 실행해야** 합니다.

### 4. 전체 파이프라인 실행

```bash
docker compose up -d --build
docker compose run producer
```

### 5. 결과 확인

| 서비스 | 주소 |
|--------|------|
| FastAPI Swagger | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

```bash
# DB 직접 조회
docker compose exec postgres psql -U prp_user -d prp_db \
  -c "SELECT model_type, count(*), round(avg(anomaly_score)::numeric,4) \
      FROM anomaly_results GROUP BY model_type;"
```

---

## 주요 설계 결정

**왜 Kafka에 이미지를 직접 넣지 않았나?**
이미지를 직접 전송하면 Kafka 메시지 크기 제한(기본 1MB)에 걸리고 처리 속도가
느려집니다. 대신 이미지는 공유 볼륨에 저장하고 Kafka에는 파일 경로만 전송합니다.
프로덕션 환경에서는 S3/MinIO에 업로드하고 URL을 전송하는 방식으로 전환합니다.

**왜 offset을 수동 commit하고 DLQ를 두나?**
`enable_auto_commit=False`로 설정해 추론이 성공한 후에만 commit합니다.
처리에 실패한 메시지는 그냥 commit해 버리면 유실되고, commit하지 않으면 동일한
메시지를 무한히 재처리(poison message)하게 됩니다. 실패 메시지를 DLQ 토픽으로
이관한 뒤 commit하여 양쪽을 모두 피합니다. DLQ 전송 자체가 실패하면 commit하지
않으므로 메시지가 보존됩니다.

**왜 memory bank에 메타데이터를 함께 저장하나?**
memory bank는 backbone, layer 조합, 입력 해상도, 카테고리에 따라 전혀 다른
객체입니다. 배열만 저장하면 설정을 바꾸고 재학습하지 않았을 때 차원이 우연히
일치하는 한 **에러 없이 잘못된 결과**가 나옵니다. 특히 카테고리만 바꾸면
다른 카테고리의 memory bank로 검사하게 됩니다. 저장 시 메타데이터를 함께 기록하고
로드 시 현재 설정과 대조해 불일치하면 즉시 중단합니다.

**왜 임계값을 config에 하드코딩하지 않나?**
기존에는 `threshold: 4.5`를 config에 적어 두었는데, 이 값은 테스트셋 정상 이미지의
최댓값에 맞춰 고른 것이라 테스트 데이터 누수였습니다. 현재는 학습용 정상 이미지의
20% holdout에서 p99로 산출해 `results/metrics/{category}.json`에 기록하고,
추론 시 그 값을 읽습니다.

**왜 model_type 컬럼을 별도로 두나?**
AutoEncoder와 PatchCore 결과를 같은 테이블에 저장하고 `model_type`으로 필터링해
성능을 비교합니다. 이 값은 config가 아니라 추론에 **실제로 사용된 모델**에서
가져옵니다.
