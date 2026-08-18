from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gasket_inspection.config import load_config, resolve_project_path  # noqa: E402
from gasket_inspection.predictor import Predictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 한 건 추론")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--front", type=Path)
    parser.add_argument("--side", type=Path)
    parser.add_argument("--combined", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["input"]["mode"] == "paired_files":
        if args.front is None or args.side is None:
            parser.error("paired_files 모드에는 --front와 --side가 모두 필요합니다.")
        if args.combined is not None:
            parser.error("paired_files 모드에서는 --combined를 함께 사용할 수 없습니다.")
    else:
        if args.combined is None:
            parser.error("combined_image 모드에는 --combined가 필요합니다.")
        if args.front is not None or args.side is not None:
            parser.error("combined_image 모드에서는 --front/--side를 함께 사용할 수 없습니다.")

    checkpoint = resolve_project_path(
        cfg, args.checkpoint if args.checkpoint is not None else cfg["inference"]["checkpoint"]
    )
    predictor = Predictor(cfg, checkpoint, device_override=args.device)
    if cfg["input"]["mode"] == "paired_files":
        result = predictor.predict_paths(args.sample_id, front_path=args.front, side_path=args.side)
    else:
        result = predictor.predict_paths(args.sample_id, combined_path=args.combined)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
