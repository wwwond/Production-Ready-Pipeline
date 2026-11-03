# PRP - Precision Realtime Pipeline

> 정상 제품 이미지만으로 학습해 불량을 찾는 제조 이상탐지 모델 + 실시간 처리 파이프라인 (개인 프로젝트, CPU 단일 환경)

---

## 결과

### MVTec AD 15개 카테고리 전체 평가 (CPU 재현)

| | Image AUROC | Pixel AUROC |
| --- | --- | --- |
| 본 구현 (PatchCore) | **0.980** | **0.978** |
| 원 논문 (Roth et al., 2022) | 0.990 | 0.981 |


### 경량화 (carpet 카테고리)

| | 메모리 뱅크 크기 | 탐지율 (정상 이미지 오탐 0건 조건) |
| --- | --- | --- |
| 전체 패치 저장 | 176 MB | 75.3 % |
| coreset 샘플링 + 주변 패치 집계 | **13 MB** | **93.3 %** |


---

## 이 프로젝트에서 내린 판단

**1. 복원 오차 방식을 버린 이유 — AUROC 0.341**
처음 구현한 CNN AutoEncoder는 AUROC 0.341로 0.5보다 낮았습니다. 불량 이미지에 정상보다 오히려 낮은 점수를 매긴다는 뜻입니다. 점수 집계 방식(평균, 최댓값 등)을 바꿔도 역전이 유지되어, 문제가 집계가 아니라 '복원 오차'라는 기준 자체에 있다고 판단했습니다. carpet처럼 복잡한 텍스처는 불량도 잘 복원되기 때문입니다. 그래서 정상 특징을 저장해 두고 거리로 판단하는 PatchCore로 교체했습니다.

**2. 평가 누수 제거 — 임계값을 테스트셋에서 정하지 않기**
참고한 코드가 불량 판정 임계값을 테스트셋 점수를 보고 정하고 있었습니다. 시험 문제를 보고 합격선을 정하는 것과 같아서 성능이 실제보다 좋게 나옵니다. 학습용 정상 이미지 일부를 떼어 낸 holdout 세트에서 임계값을 정하도록 바꾸고, 시드를 고정해 재실행해도 같은 결과가 나오게 했습니다.

**3. GPU 없이 돌리기 위한 경량화**
메모리 뱅크를 coreset 샘플링으로 대표 특징만 남기고, 주변 패치를 함께 집계해 작은 결함에 덜 민감해지는 문제를 보완했습니다. 메모리는 176 MB → 13 MB로 줄고 carpet 탐지율은 오히려 올랐습니다.

**4. Kafka에는 이미지가 아니라 경로만**
이미지를 메시지에 직접 넣으면 기본 크기 제한(1 MB)에 걸리고 느려집니다. 이미지는 공유 볼륨에 두고 Kafka에는 파일 경로만 보냅니다. 실서비스라면 오브젝트 스토리지(S3, MinIO) URL로 바꿀 부분입니다.

**5. 추론 성공 후에만 offset commit**
`enable_auto_commit=False`로 두고 추론과 저장이 끝난 뒤 commit합니다. 중간에 실패해도 메시지를 잃지 않고 다시 처리합니다.

---

## 구조

```
[이미지 폴더]
     ↓
[Kafka Producer] ── 파일 경로만 전송
     ↓
[Kafka Topic: anomaly-detection]
     ↓
[Kafka Consumer] ── 이미지 로드 → PatchCore 추론 → Anomaly Map 생성
     ↓
[PostgreSQL] ── 결과 저장 (model_type 컬럼으로 모델별 비교)
     ↓
[FastAPI] ── 추론 / 조회 / 통계 / 히트맵 엔드포인트
```

Docker Compose로 7개 서비스를 한 번에 실행합니다.

| 분류 | 기술 |
| --- | --- |
| 모델 | PatchCore (Wide ResNet50 특징), 비교용 CNN AutoEncoder |
| 파이프라인 | Kafka |
| 서빙 | FastAPI |
| 저장 | PostgreSQL |
| 배포 | Docker Compose |

---

## 한계

- 공개 데이터셋(MVTec AD) 결과이며, 실제 공정 이미지로는 검증하지 않았습니다.
- 2D 이미지만 다룹니다. 깊이·포인트 클라우드 기반 3D 검사는 범위 밖입니다.


---

## 실행

```bash
git clone https://github.com/wwwond/Production-Ready-Pipeline.git
cd Production-Ready-Pipeline
cp .env.example .env
```

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)를 내려받아 `data/mvtec/`에 둡니다.

```bash
# 모델만
python models/patchcore/train.py
python models/patchcore/inference.py

# 전체 파이프라인
docker compose up -d
docker compose run producer
```

FastAPI 문서: http://localhost:8000/docs


---

## 기록

진행 과정 상세: [Notion](https://accurate-seed-914.notion.site/PRP-372f17ee028580628350ceb1bf923c2c?source=copy_link)
