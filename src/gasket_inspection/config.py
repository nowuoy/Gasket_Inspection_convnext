from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .constants import EXPECTED_CLASS_IDS


class ConfigError(ValueError):
    """설정값이 서로 모순되거나 필수값이 없을 때 발생합니다."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("YAML 최상위 값은 mapping이어야 합니다.")

    cfg = deepcopy(loaded)
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(config_path.parent.parent)
    validate_config(cfg)
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_project_root"])


def resolve_project_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root(cfg) / path
    return path.resolve()


def class_ids(cfg: dict[str, Any]) -> list[str]:
    return [item["id"] for item in cfg["classes"]]


def class_display_names(cfg: dict[str, Any]) -> dict[str, str]:
    return {item["id"]: item["name_ko"] for item in cfg["classes"]}


def checkpoint_safe_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in cfg.items() if not key.startswith("_")}


def choose_device(requested: str) -> str:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ConfigError("device는 auto, cpu, cuda 중 하나여야 합니다.")
    if requested == "cpu":
        return "cpu"

    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda이지만 CUDA를 사용할 수 없습니다.")
    if requested == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("device는 auto, cpu, cuda 중 하나여야 합니다.")
    if not isinstance(cfg.get("seed", 42), int) or isinstance(cfg.get("seed", 42), bool):
        raise ConfigError("seed는 정수여야 합니다.")
    try:
        configured_ids = [item["id"] for item in cfg["classes"]]
        configured_names = [item["name_ko"] for item in cfg["classes"]]
    except (KeyError, TypeError) as exc:
        raise ConfigError("classes는 id와 name_ko를 가진 항목 목록이어야 합니다.") from exc

    if tuple(configured_ids) != EXPECTED_CLASS_IDS:
        raise ConfigError(
            "클래스 순서는 checkpoint 호환성을 위해 다음과 같아야 합니다: "
            + ", ".join(EXPECTED_CLASS_IDS)
        )
    if len(set(configured_names)) != len(configured_names):
        raise ConfigError("클래스 표시명이 중복되었습니다.")

    input_cfg = cfg.get("input", {})
    if int(input_cfg.get("image_size", 0)) <= 0:
        raise ConfigError("input.image_size는 양수여야 합니다.")
    for key in ("mean", "std", "padding_value"):
        value = input_cfg.get(key)
        if not isinstance(value, list) or len(value) != 3:
            raise ConfigError(f"input.{key}는 RGB 3개 값 목록이어야 합니다.")
    numeric_values = input_cfg["mean"] + input_cfg["std"]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_values
    ):
        raise ConfigError("input.mean/std는 유한한 숫자여야 합니다.")
    if any(float(value) <= 0 for value in input_cfg["std"]):
        raise ConfigError("input.std는 모두 0보다 커야 합니다.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
        for value in input_cfg["padding_value"]
    ):
        raise ConfigError("input.padding_value는 0~255 범위여야 합니다.")
    model_cfg = cfg.get("model", {})
    if model_cfg.get("architecture") not in {"convnext_tiny", "convnext_small", "convnext_base"}:
        raise ConfigError("지원하지 않는 model.architecture입니다.")
    if not 0.0 <= float(model_cfg.get("dropout", -1.0)) < 1.0:
        raise ConfigError("model.dropout은 0 이상 1 미만이어야 합니다.")

    train_cfg = cfg.get("train", {})
    positive_integer_keys = ("batch_size", "epochs", "early_stopping_patience")
    for key in positive_integer_keys:
        if int(train_cfg.get(key, 0)) <= 0:
            raise ConfigError(f"train.{key}는 양의 정수여야 합니다.")
    if int(train_cfg.get("num_workers", -1)) < 0:
        raise ConfigError("train.num_workers는 0 이상이어야 합니다.")
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", -1))
    if not 0 <= freeze_epochs < int(train_cfg["epochs"]):
        raise ConfigError("train.freeze_backbone_epochs는 전체 epochs보다 작아야 합니다.")
    for key in ("learning_rate", "backbone_learning_rate", "weight_decay"):
        if float(train_cfg.get(key, -1.0)) < 0.0:
            raise ConfigError(f"train.{key}는 0 이상이어야 합니다.")
    if float(train_cfg.get("learning_rate", 0.0)) == 0.0:
        raise ConfigError("train.learning_rate는 0보다 커야 합니다.")
    if train_cfg.get("class_weighting") not in {"auto", "none"}:
        raise ConfigError("train.class_weighting은 auto 또는 none이어야 합니다.")
    label_smoothing = float(train_cfg.get("label_smoothing", -1.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ConfigError("train.label_smoothing은 0 이상 1 미만이어야 합니다.")
    if float(train_cfg.get("severity_loss_weight", -1.0)) <= 0.0:
        raise ConfigError("train.severity_loss_weight는 0보다 커야 합니다.")
    allowed_checkpoint_metrics = {"macro_f1", "ng_recall", "severity_mae"}
    if train_cfg.get("checkpoint_metric", "macro_f1") not in allowed_checkpoint_metrics:
        raise ConfigError("train.checkpoint_metric 값이 올바르지 않습니다.")
    if train_cfg.get("checkpoint_mode", "max") not in {"max", "min"}:
        raise ConfigError("train.checkpoint_mode는 max 또는 min이어야 합니다.")
    checkpoint_metric = train_cfg.get("checkpoint_metric", "macro_f1")
    checkpoint_mode = train_cfg.get("checkpoint_mode", "max")
    expected_mode = "min" if checkpoint_metric == "severity_mae" else "max"
    if checkpoint_mode != expected_mode:
        raise ConfigError(
            f"train.checkpoint_metric={checkpoint_metric}에는 checkpoint_mode={expected_mode}를 사용하세요."
        )
    augmentation = train_cfg.get("augmentation", {})
    if not 0.0 <= float(augmentation.get("rotation_degrees", -1.0)) <= 180.0:
        raise ConfigError("augmentation.rotation_degrees 범위가 올바르지 않습니다.")
    if not 0.0 <= float(augmentation.get("translate_fraction", -1.0)) < 1.0:
        raise ConfigError("augmentation.translate_fraction 범위가 올바르지 않습니다.")
    for key in ("brightness", "contrast", "saturation"):
        if not 0.0 <= float(augmentation.get(key, -1.0)) <= 1.0:
            raise ConfigError(f"augmentation.{key}는 0~1이어야 합니다.")

    boolean_fields = (
        (model_cfg, "pretrained", "model.pretrained"),
        (train_cfg, "mixed_precision", "train.mixed_precision"),
        (train_cfg, "enforce_group_disjoint", "train.enforce_group_disjoint"),
        (cfg.get("inference", {}), "use_amp", "inference.use_amp"),
        (cfg.get("realtime", {}), "retry_errors", "realtime.retry_errors"),
    )
    for section, key, display_key in boolean_fields:
        if not isinstance(section.get(key), bool):
            raise ConfigError(f"{display_key}는 true 또는 false여야 합니다.")

    good_class = cfg["decision"].get("good_class", "good")
    if good_class not in configured_ids:
        raise ConfigError(f"decision.good_class가 클래스 목록에 없습니다: {good_class}")
    defect_ids = [class_id for class_id in configured_ids if class_id != good_class]
    thresholds = cfg["decision"].get("severity_ng_thresholds", {})
    missing = [class_id for class_id in defect_ids if thresholds.get(class_id) is None]
    if missing:
        raise ConfigError("severity threshold가 비어 있습니다: " + ", ".join(missing))
    if any(not 0.0 <= float(thresholds[class_id]) <= 1.0 for class_id in defect_ids):
        raise ConfigError("severity threshold는 모두 0~1이어야 합니다.")

    inference_cfg = cfg.get("inference", {})
    if int(inference_cfg.get("warmup_iterations", 0)) < 0:
        raise ConfigError("inference.warmup_iterations는 0 이상이어야 합니다.")

    realtime_cfg = cfg.get("realtime", {})
    if float(realtime_cfg.get("poll_interval_s", 0.0)) <= 0.0:
        raise ConfigError("realtime.poll_interval_s는 0보다 커야 합니다.")
    if int(realtime_cfg.get("stable_checks", 0)) < 1:
        raise ConfigError("realtime.stable_checks는 1 이상이어야 합니다.")
    if float(realtime_cfg.get("retry_backoff_s", -1.0)) < 0.0:
        raise ConfigError("realtime.retry_backoff_s는 0 이상이어야 합니다.")
    if int(realtime_cfg.get("max_retry_attempts", 0)) < 1:
        raise ConfigError("realtime.max_retry_attempts는 1 이상이어야 합니다.")
    regex_requirements = {"filename_regex": {"id", "ext"}}
    for key, required_groups in regex_requirements.items():
        try:
            pattern = re.compile(str(realtime_cfg[key]))
        except (KeyError, re.error) as exc:
            raise ConfigError(f"realtime.{key}가 올바른 정규식이 아닙니다.") from exc
        missing_groups = required_groups - set(pattern.groupindex)
        if missing_groups:
            raise ConfigError(
                f"realtime.{key}에 named group이 없습니다: " + ", ".join(sorted(missing_groups))
            )
