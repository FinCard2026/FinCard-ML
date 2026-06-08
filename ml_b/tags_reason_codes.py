# tags_reason_codes.py
# Section 12 기준: 정책별 tags 및 reason_codes 생성

from datetime import datetime, timezone


def generate_tags_and_reason_codes(
    policy: dict,
    user_profile: dict,
    cluster_id: int,
) -> tuple[list, list]:
    """
    정책 1개에 대한 tags / reason_codes 생성

    Args:
        policy: candidate_policies 내 정책 1개
        user_profile: final_user_profile
        cluster_id: ML-A로부터 받은 군집 ID

    Returns:
        (tags: list, reason_codes: list)
    """
    tags = []
    reason_codes = []

    category = policy.get("category", "")
    goal = user_profile.get("goal", "")
    selected_categories = user_profile.get("selected_categories", [])
    housing_type = user_profile.get("housing_type", "")
    job_type = user_profile.get("job_type", "")
    profile_completeness = user_profile.get("profile_completeness_score", 1.0)

    # --- 카테고리 태그 ---
    CATEGORY_TAG_MAP = {
        "asset_building": "저축·적금",
        "housing": "주거",
        "employment": "직업",
        "living_support": "생활비",
        "education": "교육",
        "tax": "세금·절세",
        "policy_finance": "필요 시 금융지원",
    }
    if category in CATEGORY_TAG_MAP:
        tags.append(CATEGORY_TAG_MAP[category])

    # --- reason_codes / 조건 태그 ---

    # 목표 일치
    if goal == category:
        reason_codes.append("MATCH_GOAL")

    # 군집 일치 (항상 포함)
    reason_codes.append("MATCH_CLUSTER")

    # 주거 일치
    if category == "housing" and housing_type == "monthly_rent":
        tags.append("월세청년추천")
        reason_codes.append("MATCH_HOUSING")

    # 고용형태 일치
    if category == "employment" and job_type in ["job_seeker", "unemployed", "part_time"]:
        tags.append("취준생 추천")
        reason_codes.append("MATCH_EMPLOYMENT")

    # 소득증빙 필요
    if policy.get("requires_income_proof"):
        tags.append("소득증빙필요")
        reason_codes.append("INCOME_PROOF_REQUIRED")

    # 프리랜서
    if job_type == "freelancer":
        tags.append("프리랜서 검토 가능")

    # 대학생
    if job_type == "student":
        tags.append("대학생 추천")

    # 마감 임박 (30일 이내)
    deadline_str = policy.get("deadline", "")
    if deadline_str:
        try:
            deadline = datetime.strptime(deadline_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            if 0 <= (deadline - now).days <= 30:
                tags.append("마감임박")
                reason_codes.append("DEADLINE_SOON")
        except ValueError:
            pass

    # 서류 확인 필요
    if policy.get("requires_income_proof") or category in ["housing", "asset_building"]:
        tags.append("조건확인필요")
        reason_codes.append("DOCUMENT_CHECK_REQUIRED")

    # profile 미완성
    if isinstance(profile_completeness, float) and profile_completeness < 0.6:
        tags.append("추가정보필요")
        reason_codes.append("PROFILE_INCOMPLETE")

    # 중복 제거 (순서 유지)
    tags = list(dict.fromkeys(tags))
    reason_codes = list(dict.fromkeys(reason_codes))

    return tags, reason_codes
