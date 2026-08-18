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
        quality_probability: float | None = None,
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
        if quality_probability is not None and (
            not math.isfinite(quality_probability) or not 0.0 <= quality_probability <= 1.0
        ):
            raise FloatingPointError("quality_probability가 유효하지 않습니다.")
        if severity is not None and (not math.isfinite(severity) or not 0.0 <= severity <= 1.0):
            raise FloatingPointError("severity가 유효하지 않습니다.")

        observed_class = max(self.class_ids, key=lambda key: probabilities[key])
        confidence = float(probabilities[observed_class])
        best_defect = max(
            (class_id for class_id in self.class_ids if class_id != self.good_class),
            key=lambda key: probabilities[key],
        )

        strategy = self.cfg.get("strategy", "class_only")
        if strategy == "class_only":
            status = "OK" if observed_class == self.good_class else "NG"
            defect = observed_class if status == "NG" else None
            rule = "provisional_good_vs_defect_class"
            threshold = None
            provisional = True
        elif strategy == "quality_head":
            if quality_probability is None:
                raise ValueError("quality_head 판정에 quality_probability가 없습니다.")
            threshold = float(self.cfg["quality_ng_threshold"])
            status = "NG" if quality_probability >= threshold else "OK"
            defect = (
                observed_class if observed_class != self.good_class else best_defect
            ) if status == "NG" else None
            rule = "quality_head"
            provisional = False
        elif strategy == "severity":
            if severity is None:
                raise ValueError("severity 판정에 severity 출력이 없습니다.")
            # severity가 높은데 class만 good인 모순을 fail-open으로 OK 처리하지 않습니다.
            candidate_defect = observed_class if observed_class != self.good_class else best_defect
            threshold = float(self.cfg["severity_ng_thresholds"][candidate_defect])
            status = "NG" if severity >= threshold else "OK"
            defect = candidate_defect if status == "NG" else None
            rule = "class_specific_severity"
            provisional = False
        else:
            raise ValueError(f"알 수 없는 decision.strategy={strategy}")

        minimum = self.cfg.get("min_class_confidence")
        if minimum is not None and confidence < float(minimum) and status != "NG":
            low_confidence_action = self.cfg.get("low_confidence_action", "NG")
            if low_confidence_action != "OK":
                status = low_confidence_action
                defect = best_defect if status == "NG" else None
                rule = "low_class_confidence"
                threshold = float(minimum)
                provisional = True

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
