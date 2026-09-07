# 다시찾음 AI Matching Service

분실물과 습득물 후보의 텍스트, 이미지, 속성, 날짜, 위치를 비교해
상위 후보를 반환하는 FastAPI 서비스입니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 테스트

```bash
pytest -q
```

실제 모델을 내려받지 않는 자동 테스트만 실행됩니다.

## API 실행

```bash
uvicorn main:app --reload --port 8001
```

- 상태 확인: `GET http://127.0.0.1:8001/health`
- 후보 순위: `POST http://127.0.0.1:8001/v1/rankings`
- API 문서: `http://127.0.0.1:8001/docs`

요청 예시는 `sample_request.example.json`에서 확인할 수 있습니다.

## 텍스트 필드 역할

- `description`: 사용자가 입력하거나 경찰청에서 제공한 원문
- `visualText`: 이미지와 비교하기 위해 사람이 직접 정리한 시각적 설명

현재는 LLM 전처리를 사용하지 않으며 `visualText`를 직접 입력합니다.
`visualText`는 BGE-M3 텍스트 점수에 중복 반영하지 않고 SigLIP2
이미지-텍스트 점수에만 사용합니다.

## 이미지-텍스트 비교 방향

가능한 입력에 따라 두 방향을 계산하고, 둘 다 있으면 평균을 사용합니다.

1. 검색물 `visualText`와 후보 이미지
2. 검색물 이미지와 후보 `visualText`

이미지나 `visualText`가 없거나 읽지 못하면 해당 점수만 제외하고 나머지
점수의 가중치를 자동으로 재정규화합니다.

기본 가중치는 다음과 같습니다.

```text
image       0.30
text        0.25
imageText   0.10
attribute   0.20
date        0.10
location    0.05
```

`imageText`는 동일 물품을 확정하는 값이 아니라 상위 후보 검색을 돕는
보조 점수입니다.

## 문장 하나로 후보 이미지 순위화

비교할 이미지들을 한 폴더에 넣은 뒤 실행합니다.

```bash
PYTHONPATH=. python scripts/rank_images_by_text.py \
  --image-dir "/후보/이미지/폴더" \
  --text "금색 YSL 로고가 있는 검은색 퀼팅 가죽 카드지갑" \
  --top-k 10
```

전체 결과는 기본적으로 `results/text_to_image_ranking.json`에 저장됩니다.
`rawCosine`과 API에서 사용하는 0~1 범위의 `normalizedScore`를 함께
확인할 수 있습니다.
