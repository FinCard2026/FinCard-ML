# ml_a/synthetic_data.py

import random
import pandas as pd


def generate_synthetic_profiles(n_per_cluster: int = 80, random_state: int = 42) -> pd.DataFrame:
    """
    K-Means 학습용 합성 사용자 프로필 생성.
    실제 개인정보가 없으므로 데모 목적의 synthetic data를 사용한다.
    """
    random.seed(random_state)
    rows = []

    # 0. 저소득 대학생형
    for _ in range(n_per_cluster):
        rows.append({
            "age": random.randint(19, 24),
            "job_type": random.choice(["student", "part_time"]),
            "monthly_income": random.choice([0, 300000, 500000, 800000]),
            "region": random.choice(["seoul", "gyeonggi", "other"]),
            "housing_type": random.choice(["with_family", "dormitory", "monthly_rent"]),
            "is_student": True,
            "is_job_seeker": False,
            "goal": random.choice(["education", "living_support", "asset_building"]),
            "has_part_time_income": random.choice([True, False]),
            "income_stability": random.choice(["unstable", "unknown"]),
        })

    # 1. 사회초년생 저축형
    for _ in range(n_per_cluster):
        rows.append({
            "age": random.randint(24, 31),
            "job_type": random.choice(["full_time", "contract"]),
            "monthly_income": random.choice([1800000, 2200000, 2600000, 3000000]),
            "region": random.choice(["seoul", "gyeonggi", "incheon", "other"]),
            "housing_type": random.choice(["with_family", "monthly_rent", "jeonse"]),
            "is_student": False,
            "is_job_seeker": False,
            "goal": random.choice(["asset_building", "tax"]),
            "has_income_proof": True,
            "tax_reported": True,
            "income_stability": "stable",
        })

    # 2. 프리랜서·소득불안정형
    for _ in range(n_per_cluster):
        rows.append({
            "age": random.randint(23, 34),
            "job_type": random.choice(["freelancer", "part_time", "other"]),
            "monthly_income": random.choice([800000, 1200000, 1500000, 2000000, 2500000]),
            "region": random.choice(["seoul", "gyeonggi", "other"]),
            "housing_type": random.choice(["monthly_rent", "with_family"]),
            "is_student": False,
            "is_job_seeker": False,
            "goal": random.choice(["asset_building", "living_support", "policy_finance"]),
            "has_income_proof": random.choice([True, False]),
            "tax_reported": random.choice([True, False]),
            "income_stability": "unstable",
        })

    # 3. 월세 부담형
    for _ in range(n_per_cluster):
        rows.append({
            "age": random.randint(20, 34),
            "job_type": random.choice(["student", "freelancer", "contract", "full_time", "part_time"]),
            "monthly_income": random.choice([700000, 1200000, 1800000, 2400000]),
            "region": random.choice(["seoul", "gyeonggi", "incheon"]),
            "housing_type": "monthly_rent",
            "is_student": random.choice([True, False]),
            "is_job_seeker": False,
            "goal": random.choice(["housing", "living_support"]),
            "is_independent_household": True,
            "is_homeless_or_no_house": True,
            "has_income_proof": random.choice([True, False]),
            "income_stability": random.choice(["stable", "unstable"]),
        })

    # 4. 취업준비·무소득형
    for _ in range(n_per_cluster):
        rows.append({
            "age": random.randint(20, 34),
            "job_type": random.choice(["job_seeker", "unemployed"]),
            "monthly_income": random.choice([0, 300000, 500000]),
            "region": random.choice(["seoul", "gyeonggi", "other"]),
            "housing_type": random.choice(["with_family", "monthly_rent"]),
            "is_student": False,
            "is_job_seeker": True,
            "goal": random.choice(["employment", "education", "living_support"]),
            "active_job_search": True,
            "income_stability": "unknown",
        })

    return pd.DataFrame(rows)