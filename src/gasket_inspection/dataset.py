from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import class_ids, resolve_project_path
from .constants import MANIFEST_COLUMNS
from .images import build_transform, load_view_pair


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    front_path: Path | None
    side_path: Path | None
    combined_path: Path | None
    defect_type: str
    is_ng: float
    severity: float
    lot_id: str
    capture_session: str
    split: str


def _optional_float(value: str, field_name: str, sample_id: str) -> float:
    value = value.strip()
    if not value:
        return math.nan
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{sample_id}: {field_name} 값이 숫자가 아닙니다: {value}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{sample_id}: {field_name}는 0.0~1.0이어야 합니다.")
    return parsed


def _optional_binary(value: str, field_name: str, sample_id: str) -> float:
    parsed = _optional_float(value, field_name, sample_id)
    if math.isnan(parsed):
        return parsed
    if parsed not in {0.0, 1.0}:
        raise ValueError(f"{sample_id}: {field_name}는 0 또는 1이어야 합니다.")
    return parsed


def _optional_path(base_dir: Path, value: str) -> Path | None:
    if not value.strip():
        return None
    path = Path(value.strip()).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"라벨 manifest를 찾을 수 없습니다: {manifest_path}")

    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("manifest header에 중복된 열이 있습니다.")
        missing_columns = [column for column in MANIFEST_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise ValueError("manifest 필수 열이 없습니다: " + ", ".join(missing_columns))
        for line_number, raw in enumerate(reader, start=2):
            missing_values = [column for column in MANIFEST_COLUMNS if raw.get(column) is None]
            if missing_values:
                raise ValueError(
                    f"manifest {line_number}행의 열 개수가 부족합니다: " + ", ".join(missing_values)
                )
            sample_id = raw["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"manifest {line_number}행: sample_id가 비어 있습니다.")
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    front_path=_optional_path(manifest_path.parent, raw["front_path"]),
                    side_path=_optional_path(manifest_path.parent, raw["side_path"]),
                    combined_path=_optional_path(manifest_path.parent, raw["combined_path"]),
                    defect_type=raw["defect_type"].strip(),
                    is_ng=_optional_binary(raw["is_ng"], "is_ng", sample_id),
                    severity=_optional_float(raw["severity"], "severity", sample_id),
                    lot_id=raw["lot_id"].strip(),
                    capture_session=raw["capture_session"].strip(),
                    split=raw["split"].strip().lower(),
                )
            )
    return rows


def validate_rows(rows: list[ManifestRow], cfg: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("manifest에 데이터 행이 없습니다. 촬영/라벨링 후 행을 추가하세요.")

    valid_classes = set(class_ids(cfg))
    seen_ids: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    mode = cfg["input"]["mode"]
    quality_required = bool(cfg["model"].get("quality_head_enabled", False))
    severity_required = bool(cfg["model"].get("severity_head_enabled", False))
    path_owners: dict[str, tuple[str, str]] = {}

    for row in rows:
        if row.sample_id in seen_ids:
            raise ValueError(f"sample_id가 중복되었습니다: {row.sample_id}")
        seen_ids.add(row.sample_id)
        if row.defect_type not in valid_classes:
            raise ValueError(f"{row.sample_id}: 알 수 없는 defect_type={row.defect_type}")
        if row.split not in {"train", "val", "test"}:
            raise ValueError(f"{row.sample_id}: split은 train, val, test 중 하나여야 합니다.")
        if mode == "paired_files":
            paths = [row.front_path, row.side_path]
            if any(path is None for path in paths):
                raise ValueError(f"{row.sample_id}: front_path/side_path가 모두 필요합니다.")
        else:
            paths = [row.combined_path]
            if row.combined_path is None:
                raise ValueError(f"{row.sample_id}: combined_path가 필요합니다.")
        for path in paths:
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"{row.sample_id}: 이미지가 없습니다: {path}")
        for view_name, path in (
            ("front", row.front_path),
            ("side", row.side_path),
            ("combined", row.combined_path),
        ):
            if path is None or (mode == "paired_files" and view_name == "combined"):
                continue
            if mode == "combined_image" and view_name != "combined":
                continue
            path_key = str(path).casefold()
            previous = path_owners.get(path_key)
            if previous is not None:
                raise ValueError(
                    f"이미지 경로가 중복 사용되었습니다: {path} "
                    f"({previous[0]}/{previous[1]}, {row.sample_id}/{view_name})"
                )
            path_owners[path_key] = (row.sample_id, view_name)
        if quality_required and math.isnan(row.is_ng):
            raise ValueError(f"{row.sample_id}: quality head 학습에는 is_ng 라벨이 필요합니다.")
        if severity_required and math.isnan(row.severity):
            raise ValueError(f"{row.sample_id}: severity head 학습에는 severity 라벨이 필요합니다.")
        counts[(row.split, row.defect_type)] += 1

    split_names = {row.split for row in rows}
    if not {"train", "val"}.issubset(split_names):
        raise ValueError("학습에는 train과 val split이 모두 필요합니다.")
    for split_name in ("train", "val"):
        missing_classes = [
            class_id for class_id in class_ids(cfg) if counts[(split_name, class_id)] == 0
        ]
        if missing_classes:
            raise ValueError(
                f"{split_name} split에 없는 클래스가 있습니다: " + ", ".join(missing_classes)
            )

    for field_name in ("is_ng", "severity"):
        present_count = sum(
            math.isfinite(float(getattr(row, field_name))) for row in rows
        )
        if present_count not in {0, len(rows)}:
            raise ValueError(
                f"선택 라벨 {field_name}는 전체 행을 채우거나 전체를 비워야 합니다. "
                f"현재 {present_count}/{len(rows)}행만 채워졌습니다."
            )

    train_cfg = cfg["train"]
    group_columns = train_cfg.get("group_columns", ["lot_id", "capture_session"])
    if not isinstance(group_columns, list) or not group_columns:
        raise ValueError("train.group_columns는 하나 이상의 열 목록이어야 합니다.")
    invalid_group_columns = set(group_columns) - {"lot_id", "capture_session"}
    if invalid_group_columns:
        raise ValueError(
            "지원하지 않는 group column: " + ", ".join(sorted(invalid_group_columns))
        )
    if train_cfg.get("enforce_group_disjoint", True):
        for group_column in group_columns:
            groups_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
            for row in rows:
                group = getattr(row, group_column)
                if not group:
                    raise ValueError(
                        f"{row.sample_id}: 누수 검사를 위해 {group_column} 값을 채워야 합니다."
                    )
                groups_by_split[row.split].add(group)
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = groups_by_split[left] & groups_by_split[right]
                if overlap:
                    raise ValueError(
                        f"{group_column} 데이터 누수: {left}/{right}에 동시에 존재: {sorted(overlap)}"
                    )

    return {
        "num_samples": len(rows),
        "counts": {f"{split_name}/{label}": count for (split_name, label), count in sorted(counts.items())},
    }


class GasketDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        cfg: dict[str, Any],
        *,
        training: bool,
    ) -> None:
        self.rows = rows
        self.input_cfg = cfg["input"]
        self.class_to_index = {name: index for index, name in enumerate(class_ids(cfg))}
        self.transform = build_transform(self.input_cfg, cfg["train"] if training else None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        front, side = load_view_pair(
            self.input_cfg,
            front_path=row.front_path,
            side_path=row.side_path,
            combined_path=row.combined_path,
        )
        return {
            "sample_id": row.sample_id,
            "front": self.transform(front),
            "side": self.transform(side),
            "class_target": torch.tensor(self.class_to_index[row.defect_type], dtype=torch.long),
            "quality_target": torch.tensor(row.is_ng, dtype=torch.float32),
            "severity_target": torch.tensor(row.severity, dtype=torch.float32),
        }


def build_datasets(cfg: dict[str, Any]) -> tuple[GasketDataset, GasketDataset, list[ManifestRow]]:
    manifest_path = resolve_project_path(cfg, cfg["train"]["manifest"])
    rows = read_manifest(manifest_path)
    validate_rows(rows, cfg)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    return (
        GasketDataset(train_rows, cfg, training=True),
        GasketDataset(val_rows, cfg, training=False),
        rows,
    )


def calculate_class_weights(rows: list[ManifestRow], ordered_class_ids: list[str]) -> torch.Tensor:
    train_counts = Counter(row.defect_type for row in rows if row.split == "train")
    missing = [label for label in ordered_class_ids if train_counts[label] == 0]
    if missing:
        raise ValueError("train split에 없는 클래스가 있습니다: " + ", ".join(missing))
    total = sum(train_counts.values())
    count_classes = len(ordered_class_ids)
    values = [total / (count_classes * train_counts[label]) for label in ordered_class_ids]
    return torch.tensor(values, dtype=torch.float32)
