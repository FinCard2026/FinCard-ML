# FinCard-ML

청년 맞춤형 금융·지원 정책 추천 웹앱의 ML 모듈.
K-Means 군집화로 사용자 유형을 분류하고, 카테고리 적합도·tags·reason_codes를 계산하여 Backend에 Response JSON을 반환한다.

---

## 역할 분담

| 담당 | 브랜치 | 담당 파트 | 주요 산출물 |
|------|--------|-----------|-------------|
| ML-A (이수빈) | `ml-a` | Feature Vector 인코딩 → K-Means 학습 → 군집 예측 | `cluster_id`, `cluster_name`, `similarity_score` |
| ML-B (이미지) | `ml-b` | 카테고리 적합도 계산 → tags/reason_codes 생성 → Response JSON 조립, ML병합 | `category_scores`, `policy_results`, 최종 Response |

---

## 프로젝트 구조

```
FinCard-ML/
├── ml_a/
│   ├── __init__.py
│   ├── cluster_meta.py        # 군집 ID별 이름·설명 정의
│   ├── encoder.py             # user_profile → 41차원 Feature Vector 변환
│   ├── predictor.py           # K-Means 예측 + rule-based 보정
│   ├── synthetic_data.py      # 학습용 합성 데이터 생성
│   └── train_kmeans.py        # K-Means 모델 학습 및 artifacts 저장
├── ml_b/
│   ├── __init__.py
│   ├── completeness.py        # profile_completeness_score 계산
│   ├── category_scores.py     # 카테고리 적합도 점수 계산
│   ├── tags_reason_codes.py   # tags / reason_codes 생성
│   └── response_builder.py   # 최종 Response JSON 조립
├── artifacts/
│   ├── kmeans_pipeline.joblib   # 학습된 모델 파일
│   └── cluster_summary.json     # 학습 결과 요약
├── data/
│   ├── benefit_master.csv             # 정책 마스터 데이터 (tags는 DB 칼럼 기반)
│   └── synthetic_user_profiles.csv   # 학습용 합성 유저 데이터 (500명)
├── test_ml_a.py    # ML-A 단독 테스트
├── test_ml_b.py    # ML-B 단독 테스트
└── test_ml_ab.py   # ML-A + ML-B 연동 통합 테스트
```

---

## 실행 환경

```bash
pip install pandas scikit-learn joblib
```

Python 3.9 이상 권장. 가상환경 사용 시:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pandas scikit-learn joblib
```

---

## 실행 방법

### 1. 모델 학습 (최초 1회 또는 재학습 시)

```bash
python -m ml_a.train_kmeans
```

완료 시 `artifacts/kmeans_pipeline.joblib`, `artifacts/cluster_summary.json` 저장됨.

### 2. 테스트 실행

```bash
# ML-A 단독 (군집 예측)
python test_ml_a.py

# ML-B 단독 (추천 결과 가공)
python test_ml_b.py

# ML-A + ML-B 연동 통합
python test_ml_ab.py
```

---

## 데이터 흐름

```
[Backend]
  final_user_profile + candidate_policies
          │
          ▼
[ML-A]  predict_cluster(user_profile)
  encode (41차원) → StandardScaler → KMeans → rule-based 보정
          │
          │  cluster_result = {
          │    "cluster_id": int,
          │    "cluster_name": str,
          │    "cluster_description": str,
          │    "similarity_score": float
          │  }
          ▼
[ML-B]  build_response(request_id, user_profile, candidate_policies, cluster_result)
  1. profile_completeness_score 계산
  2. category_scores 계산 (5개 카테고리, rank 정렬)
  3. top_category 기준 정책 1개 선택  ← v3.0 spec: top 1 반환
  4. tags + reason_codes 생성
  5. Response JSON 반환
          │
          ▼
[Backend → Frontend]
  카테고리 적합도 순 정렬 / Azure OpenAI 자연어 생성 / 카드 UI 출력
```

> **정책 필터링은 Backend 담당.** ML은 Backend가 1차 필터링한 `candidate_policies`만 받아 처리한다.

---

## ML-A 상세

### Feature Vector 구성 (총 41차원)

| 그룹 | 항목 | 처리 방식 |
|------|------|-----------|
| 수치형 | `age`, `monthly_income` | income: 원 → 만원 변환 후 스케일링 |
| job_type | student / freelancer / job_seeker / part_time / contract / full_time / unemployed / other | One-Hot (8차원) |
| housing_type | with_family / monthly_rent / jeonse / own / dormitory / other / unknown | One-Hot (7차원) |
| region | all / seoul / gyeonggi / incheon / busan / other | One-Hot (6차원) |
| goal | housing / employment / asset_building / living_support / education / tax / policy_finance / other | One-Hot (8차원) |
| boolean | is_student, is_job_seeker, has_income_proof, tax_reported, is_independent_household, is_homeless_or_no_house, income_stability_stable, income_stability_unstable, active_job_search, has_part_time_income | True→1 / False·unknown·None→0 (10차원) |

> **인코딩 주의:** job_type·housing_type·goal은 명목형 변수이므로 정수 인코딩 대신 One-Hot 인코딩 적용. 정수 인코딩 시 K-Means 거리 계산에서 의미 없는 순서 관계가 발생함.

### 군집 정의

| cluster_id | cluster_name | 주요 특징 |
|:---:|------|------|
| 0 | 저소득 대학생형 | 소득 낮음, 대학생·아르바이트 비중 높음 |
| 1 | 사회초년생 저축형 | 정규직·계약직, 자산형성 관심 높음 |
| 2 | 프리랜서·소득불안정형 | 프리랜서, 소득 증빙 여부가 추천 정확도에 영향 |
| 3 | 월세 부담형 | 월세 거주, 주거지원 니즈 큼 |
| 4 | 취업준비·무소득형 | 구직자·무직, 취업·생활지원 니즈 큼 |

### Rule-based 보정

K-Means 예측 후 아래 조건을 우선순위 순서대로 평가하여 해당하면 `cluster_id`를 덮어씀.

| 우선순위 | 조건 | 보정 cluster_id |
|:---:|------|:---:|
| 1 | job_type ∈ {job_seeker, unemployed} | 4 |
| 2 | housing_type == monthly_rent AND goal == housing | 3 |
| 3 | job_type == student AND monthly_income ≤ 1,000,000 | 0 |
| 4 | job_type == freelancer AND income_stability == unstable | 2 |
| 5 | job_type ∈ {full_time, contract} AND goal ∈ {asset_building, tax} | 1 |
| — | 해당 없음 | K-Means 결과 유지 |

---

## ML-B 상세

### category_scores 계산 공식

```
Category Score = 0.40 × Cluster Category Preference
               + 0.35 × Goal Match
               + 0.25 × Profile Completeness
```

- **Cluster Category Preference:** 해당 군집에서 해당 카테고리의 선호도 (현재: 임의값. K-Means 실 데이터 학습 완료 후 군집 분포 기반으로 교체 필요)
- **Goal Match:** goal 완전 일치 `1.0` / selected_categories 포함 `0.5` / 없음 `0.0`
- **Profile Completeness:** 아래 기준의 `profile_completeness_score` 사용
- 5개 카테고리 전체 계산 → score 내림차순 정렬 → rank 부여

### profile_completeness_score

사용자 유형별 필수 항목 중 실제 응답된 비율 (범위: 0.0 ~ 1.0).
`unknown`, `None`, `-1`, `""` 는 미응답으로 처리.

| 유형 조건 | 필수 항목 |
|-----------|-----------|
| job_type == freelancer | has_income_proof, tax_reported, income_stability |
| housing_type == monthly_rent | is_independent_household, is_homeless_or_no_house, rent_amount_known |
| job_type ∈ {job_seeker, unemployed} | active_job_search, previous_employment_program |
| job_type == student | has_part_time_income, education_support_interest, housing_support_interest |
| job_type == part_time | has_part_time_income, income_stability |

job_type과 housing_type이 동시에 해당하면 필수 항목 **합집합**으로 계산.

### tags 구조

`tags`는 `benefit_master.csv`의 DB 칼럼(`tags_situation`, `tags_condition`, `tags_benefit`)에서 읽어온 값을 사용한다. (v3.0 변경: 코드 생성 방식 → DB 드리븐 방식)

| 분류 | 예시 |
|------|------|
| 카테고리 | 저축·적금 / 주거 / 직업 / 생활비 / 교육 / 세금·절세 |
| 조건 | 소득증빙필요 / 무주택조건 / 월세청년추천 / 프리랜서 검토 가능 / 마감임박 |
| 지식수준 | 초보자도 이해 쉬움 / 조건이 복잡해요 / 서류 확인 필요 |
| profile | 추가정보필요 / 추천정확도상승가능 |

### reason_codes 생성 조건

| 코드 | 발생 조건 |
|------|-----------|
| `MATCH_GOAL` | 정책 category == 사용자 goal |
| `MATCH_CLUSTER` | 항상 포함 |
| `MATCH_HOUSING` | 정책 category == housing AND housing_type == monthly_rent |
| `MATCH_EMPLOYMENT` | 정책 category == employment AND job_type ∈ {job_seeker, unemployed, part_time} |
| `LOW_INCOME_SUPPORT` | monthly_income < 1,500,000 AND 정책이 저소득 지원 성격 |
| `INCOME_PROOF_REQUIRED` | 정책의 requires_income_proof == True |
| `DEADLINE_SOON` | 정책 마감일까지 30일 이내 |
| `DOCUMENT_CHECK_REQUIRED` | requires_income_proof == True 또는 category ∈ {housing, asset_building} |
| `BEGINNER_FRIENDLY` | financial_literacy_level == beginner |
| `PROFILE_INCOMPLETE` | profile_completeness_score < 0.6 |

---

## ML-A ↔ ML-B 연동 인터페이스

```python
# ML-A
from ml_a.predictor import predict_cluster

cluster_result = predict_cluster(user_profile)
# 반환값:
# {
#   "cluster_id": int,           # 0~4
#   "cluster_name": str,
#   "cluster_description": str,
#   "similarity_score": float    # 0~1
# }

# ML-B
from ml_b.response_builder import build_response

response = build_response(
    request_id=request_id,
    user_profile=user_profile,
    candidate_policies=candidate_policies,
    cluster_result=cluster_result,
)
```

---

## Response JSON 구조

전체 응답 구조 (명세서 v3.0 Section 11 기준):

```json
{
  "request_id": "req_20260530_002",
  "model_version": "kmeans_v1.1",
  "user_profile_result": {
    "cluster_id": 2,
    "cluster_name": "프리랜서·소득불안정형",
    "cluster_description": "프리랜서 또는 아르바이트 비중이 높고, 소득 증빙 여부가 추천 정확도에 큰 영향을 주는 유형입니다.",
    "similarity_score": 0.86,
    "financial_literacy_level": "beginner",
    "profile_completeness_score": 0.78,
    "top_category": "asset_building",
    "recommended_policy_count": 2
  },
  "category_scores": [
    { "category": "asset_building", "score": 0.84, "rank": 1 },
    { "category": "housing",        "score": 0.78, "rank": 2 },
    { "category": "living_support", "score": 0.63, "rank": 3 }
  ],
  "policy_results": [
    {
      "policy_id": "p001",
      "policy_name": "청년내일저축계좌",
      "category": "asset_building",
      "tags": ["저축·적금", "소득증빙필요", "프리랜서 검토 가능", "초보자도 이해 쉬움"],
      "reason_codes": ["MATCH_GOAL", "MATCH_CLUSTER", "INCOME_PROOF_REQUIRED", "BEGINNER_FRIENDLY"]
    }
  ]
}
```

- `policy_results`는 **top_category 기준 정책 1개만 반환** (v3.0 spec)
- `similarity_score`는 ML-A 반환값이 있는 경우에만 포함
- 정책 노출 순서 결정 및 자연어 설명 생성(reason_codes → 문장)은 **Backend/Azure OpenAI 담당**

### MVP 최소 응답 (연동 이슈 발생 시 즉시 전환)

```json
{
  "request_id": "req_20260530_002",
  "model_version": "kmeans_v1.1",
  "cluster_id": 2,
  "cluster_name": "프리랜서·소득불안정형",
  "profile_completeness_score": 0.78,
  "top_category": "asset_building",
  "category_scores": [
    { "category": "asset_building", "score": 0.84, "rank": 1 },
    { "category": "housing",        "score": 0.78, "rank": 2 }
  ],
  "policy_results": [
    {
      "policy_id": "p001",
      "tags": ["저축·적금", "소득증빙필요"],
      "reason_codes": ["MATCH_CLUSTER", "INCOME_PROOF_REQUIRED"]
    }
  ]
}
```

생략 가능 필드: `similarity_score`, `cluster_description`, `policy_name`, `category` (policy_results 내)

---

## 에러 응답

```json
{
  "request_id": "req_20260530_002",
  "status": "error",
  "error_code": "INVALID_INPUT",
  "message": "monthly_income 값이 누락되었습니다."
}
```

| error_code | 의미 |
|------------|------|
| `INVALID_INPUT` | 필수 입력값 누락 또는 형식 오류 |
| `MODEL_NOT_READY` | 모델 로딩 실패 (`artifacts/` 디렉토리 확인) |
| `EMPTY_POLICY_LIST` | candidate_policies가 비어 있음 |
| `UNKNOWN_CATEGORY` | 정의되지 않은 카테고리 입력 |
| `INTERNAL_ERROR` | 기타 내부 오류 |

---

## 현재 한계 및 교체 필요 항목

| 항목 | 현재 상태 | 교체 조건 |
|------|-----------|-----------|
| `CLUSTER_CATEGORY_PREFERENCE` | 임의 지정값 | K-Means 실 데이터 학습 완료 후 군집 분포 기반으로 교체 |
| K-Means 학습 데이터 | 합성 데이터 500명 | 실 사용자 데이터 수집 후 재학습 |
| `benefit_master.csv` | 수동 작성 (일부 행 미완성) | 실 정책 DB 연동 시 교체 / tags 칼럼 완성 필요 |
| tags 생성 방식 | DB 칼럼 기반 (tags_situation, tags_condition, tags_benefit) | benefit_master 데이터 완성 시 자동 적용 |
