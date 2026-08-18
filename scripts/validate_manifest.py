from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gasket_inspection.config import load_config, resolve_project_path  # noqa: E402
from gasket_inspection.dataset import read_manifest, validate_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨 manifest 및 이미지 경로 검증")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = resolve_project_path(cfg, cfg["train"]["manifest"])
    summary = validate_rows(read_manifest(manifest), cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

