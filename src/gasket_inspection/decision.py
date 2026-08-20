from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InspectionDecision:
    status: str
    defect_type: str | None
    defect_name_ko: str | None
    observed_class: str
    observed_class_name_ko: str
    confidence: float
    defect_confidence: float | None
    applied_rule: str
    applied_threshold: float | None
    criteria_version: str
    provisional: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionPolicy:
    """신경망 출력과 현장 OK/NG 기준을 분리합니다."""

    def __init__(
        self,
        decision_cfg: dict[str, Any],
        ordered_class_ids: list[str],
        display_names: dict[str, str],
    ) -> None:
        self.cfg = decision_cfg
        self.class_ids = ordered_class_ids
        self.display_names = display_names
        self.good_class = decision_cfg.get("good_class", "good")
        if self.good_class not in ordered_class_ids:
            raise ValueError(f"good_class가 클래스 목록에 없습니다: {self.good_class}")

    def decide(
        self,
        probabilities: dict[str, float],
        *,
        severity: float | None = None,
    ) -> InspectionDecision:
        missing = set(self.class_ids) - set(probabilities)
        if missing:
            raise ValueError("확률이 누락된 클래스: " + ", ".join(sorted(missing)))
        invalid = [
            class_id
            for class_id in self.class_ids
            if not math.isfinite(float(probabilities[class_id]))
            or not 0.0 <= float(probabilities[class_id]) <= 1.0
        ]
        if invalid:
            raise FloatingPointError("유효하지 않은 클래스 확률: " + ", ".join(invalid))
        probability_sum = sum(float(probabilities[class_id]) for class_id in self.class_ids)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise FloatingPointError(f"클래스 확률 합이 1이 아닙니다: {probability_sum}")
        if severity is None:
            raise ValueError("severity 판정에 severity 출력이 없습니다.")
        if not math.isfinite(severity) or not 0.0 <= severity <= 1.0:
            raise FloatingPointError("severity가 유효하지 않습니다.")

        observed_class = max(self.class_ids, key=lambda key: probabilities[key])
        confidence = float(probabilities[observed_class])
        best_defect = max(
            (class_id for class_id in self.class_ids if class_id != self.good_class),
            key=lambda key: probabilities[key],
        )

        # class head는 적용할 결함별 threshold와 표시할 불량 종류를 선택합니다.
        # severity가 높은데 class만 good인 경우에도 fail-open으로 OK 처리하지 않습니다.
        candidate_defect = observed_class if observed_class != self.good_class else best_defect
        threshold = float(self.cfg["severity_ng_thresholds"][candidate_defect])
        status = "NG" if severity >= threshold else "OK"
        defect = candidate_defect if status == "NG" else None
        rule = "class_specific_severity"
        provisional = str(self.cfg.get("criteria_version", "TODO")).upper().startswith("TODO")

        return self._make(
            status=status,
            defect=defect,
            observed_class=observed_class,
            confidence=confidence,
            defect_confidence=float(probabilities[defect]) if defect is not None else None,
            rule=rule,
            threshold=threshold,
            provisional=provisional,
        )

    def _make(
        self,
        *,
        status: str,
        defect: str | None,
        observed_class: str,
        confidence: float,
        defect_confidence: float | None,
        rule: str,
        threshold: float | None,
        provisional: bool,
    ) -> InspectionDecision:
        return InspectionDecision(
            status=status,
            defect_type=defect,
            defect_name_ko=self.display_names[defect] if defect is not None else None,
            observed_class=observed_class,
            observed_class_name_ko=self.display_names[observed_class],
            confidence=confidence,
            defect_confidence=defect_confidence,
            applied_rule=rule,
            applied_threshold=threshold,
            criteria_version=str(self.cfg.get("criteria_version", "unknown")),
            provisional=provisional,
        )
