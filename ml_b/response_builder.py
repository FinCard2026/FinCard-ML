# response_builder.py
# Section 11 명세 기준: ML → Backend Response JSON 조립
# 정책은 category_scores 최상위 카테고리 기준 1개만 반환

from ml_b.completeness import calc_completeness
from ml_b.category_scores import calc_category_scores
from ml_b.tags_reason_codes import generate_tags_and_reason_codes

# Section 9 군집 정의
CLUSTER_META = {
    0: {
        "cluster_name": "저소득 대학생형",
        "cluster_description": "소득이 낮고 대학생/아르바이트 비중이 높은 유형입니다.",
    },
    1: {
        "cluster_name": "사회초년생 저축형",
        "cluster_description": "정규직·계약직 비중이 높고 자산형성 관심이 높은 유형입니다.",
    },
    2: {
        "cluster_name": "프리랜서·소득불안정형",
        "cluster_description": "프리랜서 또는 아르바이트 비중이 높고, 소득 증빙 여부가 추천 정확도에 큰 영향을 주는 유형입니다.",
    },
    3: {
        "cluster_name": "월세 부담형",
        "cluster_description": "월세 거주 비중이 높고 주거지원 니즈가 큰 유형입니다.",
    },
    4: {
        "cluster_name": "취업준비·무소득형",
        "cluster_description": "구직자/무직 비중이 높고 취업·생활지원 니즈가 큰 유형입니다.",
    },
}


def build_response(
    request_id: str,
    user_profile: dict,
    candidate_policies: list,
    cluster_result: dict,
) -> dict:
    """
    ML-B 최종 Response JSON 생성

    Args:
        request_id: 요청 식별자
        user_profile: final_user_profile (Section 6-3)
        candidate_policies: Backend에서 1차 필터링된 정책 목록 (Section 5)
        cluster_result: ML-A로부터 받은 군집 결과
            {
                "cluster_id": int,
                "similarity_score": float  # optional
            }

    Returns:
        Section 11 명세 구조의 dict
    """
    cluster_id = cluster_result.get("cluster_id", 2)  # 기본값: 프리랜서형 (mock)
    similarity_score = cluster_result.get("similarity_score", None)
    cluster_meta = CLUSTER_META.get(cluster_id, {})

    # 완성도 계산
    completeness = calc_completeness(user_profile)

    # profile에 completeness 주입 (tags_reason_codes에서 참조)
    user_profile["profile_completeness_score"] = completeness

    # 카테고리 적합도 계산
    category_scores = calc_category_scores(
        cluster_id=cluster_id,
        goal=user_profile.get("goal", ""),
        completeness_score=completeness,
        selected_categories=user_profile.get("selected_categories", []),
    )

    # 최상위 카테고리 추출
    top_category = category_scores[0]["category"] if category_scores else ""

    # 최상위 카테고리 정책 1개 선택
    top_policy = next(
        (p for p in candidate_policies if p.get("category") == top_category),
        candidate_policies[0] if candidate_policies else None,
    )

    # policy_results 구성
    policy_results = []
    if top_policy:
        tags, reason_codes = generate_tags_and_reason_codes(
            policy=top_policy,
            user_profile=user_profile,
            cluster_id=cluster_id,
        )
        policy_results.append({
            "policy_id": top_policy.get("policy_id"),
            "policy_name": top_policy.get("policy_name"),
            "category": top_policy.get("category"),
            "tags": tags,
            "reason_codes": reason_codes,
        })

    # user_profile_result 구성
    user_profile_result = {
        "cluster_id": cluster_id,
        "cluster_name": cluster_meta.get("cluster_name", ""),
        "cluster_description": cluster_meta.get("cluster_description", ""),
        "financial_literacy_level": user_profile.get("financial_literacy_level", ""),
        "profile_completeness_score": completeness,
        "top_category": top_category,
        "recommended_policy_count": len(candidate_policies),
    }
    if similarity_score is not None:
        user_profile_result["similarity_score"] = similarity_score

    return {
        "request_id": request_id,
        "model_version": "kmeans_v1.1",
        "user_profile_result": user_profile_result,
        "category_scores": category_scores,
        "policy_results": policy_results,
    }
