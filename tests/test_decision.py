from __future__ import annotations

import math

import pytest

from gasket_inspection.decision import DecisionPolicy


CLASSES = [
    "good",
    "shrinkage",
    "thread_defect",
    "incomplete_molding",
    "burr",
    "contamination",
]
NAMES = {class_id: class_id for class_id in CLASSES}


def probabilities(**overrides: float) -> dict[str, float]:
    values = {class_id: 0.02 for class_id in CLASSES}
    values.update(overrides)
    total = sum(values.values())
    return {class_id: value / total for class_id, value in values.items()}


def test_class_only_returns_ng_with_exact_defect() -> None:
    policy = DecisionPolicy(
        {
            "strategy": "class_only",
            "good_class": "good",
            "criteria_version": "test-v1",
            "min_class_confidence": None,
        },
        CLASSES,
        NAMES,
    )
    decision = policy.decide(probabilities(burr=0.82))
    assert decision.status == "NG"
    assert decision.defect_type == "burr"
    assert decision.provisional is True


def test_class_only_returns_ok_for_good() -> None:
    policy = DecisionPolicy(
        {"strategy": "class_only", "good_class": "good", "min_class_confidence": None},
        CLASSES,
        NAMES,
    )
    decision = policy.decide(probabilities(good=0.90))
    assert decision.status == "OK"
    assert decision.defect_type is None


def test_low_confidence_fails_safe_to_ng() -> None:
    policy = DecisionPolicy(
        {
            "strategy": "class_only",
            "good_class": "good",
            "min_class_confidence": 0.80,
            "low_confidence_action": "NG",
        },
        CLASSES,
        NAMES,
    )
    decision = policy.decide(probabilities(good=0.40, contamination=0.35))
    assert decision.status == "NG"
    assert decision.defect_type == "contamination"
    assert decision.applied_rule == "low_class_confidence"


def test_severity_threshold_boundary_is_ng() -> None:
    policy = DecisionPolicy(
        {
            "strategy": "severity",
            "good_class": "good",
            "min_class_confidence": None,
            "severity_ng_thresholds": {
                "shrinkage": 0.60,
                "thread_defect": 0.60,
                "incomplete_molding": 0.60,
                "burr": 0.60,
                "contamination": 0.60,
            },
        },
        CLASSES,
        NAMES,
    )
    assert policy.decide(probabilities(burr=0.90), severity=0.60).status == "NG"
    minor = policy.decide(probabilities(burr=0.90), severity=0.59)
    assert minor.status == "OK"
    assert minor.observed_class == "burr"


def test_nan_probability_never_becomes_ok() -> None:
    policy = DecisionPolicy(
        {"strategy": "class_only", "good_class": "good", "min_class_confidence": None},
        CLASSES,
        NAMES,
    )
    invalid = probabilities(good=0.9)
    invalid["burr"] = math.nan
    with pytest.raises(FloatingPointError):
        policy.decide(invalid)
