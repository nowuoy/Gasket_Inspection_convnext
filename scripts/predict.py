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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = resolve_project_path(
        cfg, args.checkpoint if args.checkpoint is not None else cfg["inference"]["checkpoint"]
    )
    predictor = Predictor(cfg, checkpoint, device_override=args.device)
    result = predictor.predict_paths(args.sample_id, image_path=args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
