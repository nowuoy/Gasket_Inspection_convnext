from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gasket_inspection.config import load_config  # noqa: E402
from gasket_inspection.training import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="단일 이미지 ConvNeXt 학습")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()
    checkpoint = train(load_config(args.config))
    print(f"최적 체크포인트: {checkpoint}")


if __name__ == "__main__":
    main()
