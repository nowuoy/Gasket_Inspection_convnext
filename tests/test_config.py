from __future__ import annotations

from pathlib import Path

from gasket_inspection.config import class_ids, load_config


def test_default_config_has_fixed_class_order() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    assert class_ids(cfg) == [
        "good",
        "shrinkage",
        "thread_defect",
        "incomplete_molding",
        "burr",
        "contamination",
    ]

