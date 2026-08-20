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


def severity_config(**overrides):
    config = {
        "good_class": "good",
        "criteria_version": "test-v1",
        "severity_ng_thresholds": {
            "shrinkage": 0.60,
            "thread_defect": 0.60,
            "incomplete_molding": 0.60,
            "burr": 0.60,
            "contamination": 0.60,
        },
    }
    config.update(overrides)
    return config


def test_severity_threshold_boundary_returns_ng() -> None:
    policy = DecisionPolicy(
        severity_config(),
        CLASSES,
        NAMES,
    )
    assert policy.decide(probabilities(burr=0.90), severity=0.60).status == "NG"
    minor = policy.decide(probabilities(burr=0.90), severity=0.59)
    assert minor.status == "OK"
    assert minor.observed_class == "burr"


def test_nan_probability_never_becomes_ok() -> None:
    policy = DecisionPolicy(
        severity_config(),
        CLASSES,
        NAMES,
    )
    invalid = probabilities(good=0.9)
    invalid["burr"] = math.nan
    with pytest.raises(FloatingPointError):
        policy.decide(invalid, severity=0.1)


def test_missing_severity_is_rejected() -> None:
    policy = DecisionPolicy(severity_config(), CLASSES, NAMES)
    with pytest.raises(ValueError, match="severity"):
        policy.decide(probabilities(good=0.9))
