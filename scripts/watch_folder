from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gasket_inspection.config import load_config, resolve_project_path  # noqa: E402
from gasket_inspection.predictor import Predictor  # noqa: E402
from gasket_inspection.realtime import FolderMonitor, SingleInstanceLock  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="카메라 저장 폴더 자동 감시/추론")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--once", action="store_true", help="현재 파일만 안정화 확인 후 처리하고 종료")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = resolve_project_path(
        cfg, args.checkpoint if args.checkpoint is not None else cfg["inference"]["checkpoint"]
    )
    state_db = resolve_project_path(cfg, cfg["realtime"]["state_db"])
    early_lock = SingleInstanceLock(state_db.with_suffix(state_db.suffix + ".lock"))
    monitor = None
    try:
        predictor = Predictor(cfg, checkpoint, device_override=args.device)
        monitor = FolderMonitor(cfg, predictor, instance_lock=early_lock)
        if args.once:
            print(f"처리 성공 건수: {monitor.run_current_files()}")
            return
        try:
            monitor.run_forever()
        except KeyboardInterrupt:
            print("\n감시를 종료합니다.")
    finally:
        if monitor is not None:
            monitor.close()
        else:
            early_lock.close()


if __name__ == "__main__":
    main()
