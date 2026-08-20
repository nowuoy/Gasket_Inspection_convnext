from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .config import choose_device, class_display_names, class_ids
from .decision import DecisionPolicy
from .images import build_transform, open_rgb, open_rgb_bytes
from .model import build_model


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    # 체크포인트는 반드시 이 프로젝트에서 직접 만든 신뢰 가능한 파일만 사용하세요.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 구버전 PyTorch 호환
        return torch.load(path, map_location=device)


class Predictor:
    def __init__(
        self,
        cfg: dict[str, Any],
        checkpoint_path: str | Path,
        *,
        device_override: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(choose_device(device_override or str(cfg.get("device", "auto"))))
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"체크포인트를 찾을 수 없습니다: {checkpoint_path}\n"
                "먼저 라벨을 채우고 학습을 실행하거나 inference.checkpoint를 수정하세요."
            )
        checkpoint = _load_checkpoint(checkpoint_path, self.device)
        required = {"model_state", "model_config", "input_config", "class_ids", "task_type"}
        missing = required - set(checkpoint)
        if missing:
            raise ValueError("호환되지 않는 체크포인트입니다. 누락: " + ", ".join(sorted(missing)))
        if checkpoint["task_type"] != "single_image_severity":
            raise ValueError("severity 전용 단일 이미지 체크포인트가 아닙니다. 다시 학습하세요.")

        configured_classes = class_ids(cfg)
        if list(checkpoint["class_ids"]) != configured_classes:
            raise ValueError("현재 설정과 체크포인트의 클래스 순서가 다릅니다.")
        checkpoint_input = checkpoint["input_config"]
        for key in ("image_size", "padding_value", "mean", "std"):
            if checkpoint_input.get(key) != cfg["input"].get(key):
                raise ValueError(f"현재 input.{key}가 학습 체크포인트와 다릅니다.")
        model_cfg = deepcopy(checkpoint["model_config"])
        model_cfg["pretrained"] = False  # 로컬 state를 읽으므로 ImageNet 파일을 다시 받지 않습니다.
        self.model = build_model(len(configured_classes), model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.transform = build_transform(checkpoint_input)
        self.class_ids = configured_classes
        self.policy = DecisionPolicy(cfg["decision"], configured_classes, class_display_names(cfg))
        self.use_amp = bool(cfg.get("inference", {}).get("use_amp", True)) and self.device.type == "cuda"
        self.model_version = _hash_file(checkpoint_path)[:12]
        self.architecture = str(model_cfg["architecture"])
        self.checkpoint_path = checkpoint_path

        warmup_iterations = int(cfg.get("inference", {}).get("warmup_iterations", 0))
        if warmup_iterations > 0:
            size = int(checkpoint_input["image_size"])
            dummy = torch.zeros((1, 3, size, size), device=self.device)
            with torch.inference_mode():
                for _ in range(warmup_iterations):
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    def predict_paths(
        self,
        sample_id: str,
        *,
        image_path: str | Path,
    ) -> dict[str, Any]:
        return self.predict_image(sample_id, open_rgb(image_path))

    def predict_image(
        self,
        sample_id: str,
        image: Image.Image,
    ) -> dict[str, Any]:
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            output = self.model(image_tensor)
            for output_name, tensor in output.items():
                if not bool(torch.isfinite(tensor).all()):
                    raise FloatingPointError(f"모델 출력에 NaN/Inf가 있습니다: {output_name}")
            probability_tensor = torch.softmax(output["class_logits"].float(), dim=1)[0]
            severity = float(output["severity"][0].item())
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        probabilities = {
            class_id: float(probability_tensor[index].item())
            for index, class_id in enumerate(self.class_ids)
        }
        decision = self.policy.decide(
            probabilities,
            severity=severity,
        )
        return {
            "schema_version": 1,
            "sample_id": sample_id,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "architecture": self.architecture,
                "checkpoint_sha256_prefix": self.model_version,
            },
            "probabilities": probabilities,
            "severity": severity,
            "decision": decision.to_dict(),
            "latency_ms": {"inference": latency_ms},
        }

    def predict_bytes(
        self,
        sample_id: str,
        *,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        image = open_rgb_bytes(image_bytes, "image bytes")
        return self.predict_image(sample_id, image)
