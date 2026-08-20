from __future__ import annotations

import hashlib
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    checkpoint_safe_config,
    choose_device,
    class_display_names,
    class_ids,
    resolve_project_path,
)
from .dataset import build_datasets, calculate_class_weights
from .decision import DecisionPolicy
from .model import SingleImageConvNeXt, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.2 계열 호환
        return torch.cuda.amp.GradScaler(enabled=enabled)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    class_loss_fn: nn.Module,
    train_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    class_loss = class_loss_fn(output["class_logits"], batch["class_target"])
    total = class_loss
    parts = {"class": float(class_loss.detach().item())}

    if "quality_logit" in output:
        target = batch["quality_target"]
        mask = torch.isfinite(target)
        if not bool(mask.any()):
            raise ValueError("quality head가 켜졌지만 현재 batch에 is_ng 라벨이 없습니다.")
        quality_loss = F.binary_cross_entropy_with_logits(output["quality_logit"][mask], target[mask])
        total = total + float(train_cfg.get("quality_loss_weight", 0.5)) * quality_loss
        parts["quality"] = float(quality_loss.detach().item())

    if "severity" in output:
        target = batch["severity_target"]
        mask = torch.isfinite(target)
        if not bool(mask.any()):
            raise ValueError("severity head가 켜졌지만 현재 batch에 severity 라벨이 없습니다.")
        severity_loss = F.smooth_l1_loss(output["severity"][mask], target[mask])
        total = total + float(train_cfg.get("severity_loss_weight", 0.25)) * severity_loss
        parts["severity"] = float(severity_loss.detach().item())

    parts["total"] = float(total.detach().item())
    return total, parts


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in ("image", "class_target", "quality_target", "severity_target"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


@torch.inference_mode()
def evaluate(
    model: SingleImageConvNeXt,
    loader: DataLoader,
    device: torch.device,
    ordered_classes: list[str],
    class_loss_fn: nn.Module,
    train_cfg: dict[str, Any],
    decision_cfg: dict[str, Any],
    display_names: dict[str, str],
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    provided_quality: list[float] = []
    probability_rows: list[list[float]] = []
    quality_predictions: list[float] = []
    severity_predictions: list[float] = []
    severity_targets: list[float] = []
    losses: list[float] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(batch["image"])
            loss, _ = _compute_loss(output, batch, class_loss_fn, train_cfg)
        pred = output["class_logits"].argmax(dim=1)
        batch_probabilities = torch.softmax(output["class_logits"].float(), dim=1)
        batch_size = int(batch_probabilities.shape[0])
        targets.extend(batch["class_target"].cpu().tolist())
        predictions.extend(pred.cpu().tolist())
        probability_rows.extend(batch_probabilities.float().cpu().tolist())
        provided_quality.extend(batch["quality_target"].cpu().tolist())
        severity_targets.extend(batch["severity_target"].cpu().tolist())
        if "quality_logit" in output:
            quality_predictions.extend(torch.sigmoid(output["quality_logit"]).float().cpu().tolist())
        else:
            quality_predictions.extend([math.nan] * batch_size)
        if "severity" in output:
            severity_predictions.extend(output["severity"].float().cpu().tolist())
        else:
            severity_predictions.extend([math.nan] * batch_size)
        losses.append(float(loss.item()))

    labels = list(range(len(ordered_classes)))
    macro_f1 = float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0))
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=ordered_classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels).tolist()

    good_index = ordered_classes.index("good")
    true_ng: list[bool] = []
    for class_target, quality, target_severity in zip(
        targets, provided_quality, severity_targets, strict=True
    ):
        if math.isfinite(quality):
            true_ng.append(bool(int(quality)))
        elif decision_cfg.get("strategy") == "severity":
            target_class = ordered_classes[class_target]
            if target_class == "good":
                true_ng.append(False)
            else:
                threshold = float(decision_cfg["severity_ng_thresholds"][target_class])
                true_ng.append(float(target_severity) >= threshold)
        else:
            true_ng.append(class_target != good_index)
    policy = DecisionPolicy(decision_cfg, ordered_classes, display_names)
    statuses: list[str] = []
    for values, quality, severity in zip(
        probability_rows, quality_predictions, severity_predictions, strict=True
    ):
        probabilities = {
            class_id: float(values[index]) for index, class_id in enumerate(ordered_classes)
        }
        statuses.append(
            policy.decide(
                probabilities,
                quality_probability=float(quality) if math.isfinite(quality) else None,
                severity=float(severity) if math.isfinite(severity) else None,
            ).status
        )
    false_accept = sum(
        truth and status == "OK" for truth, status in zip(true_ng, statuses, strict=True)
    )
    false_reject = sum(
        (not truth) and status == "NG" for truth, status in zip(true_ng, statuses, strict=True)
    )
    true_positive_ng = sum(
        truth and status == "NG" for truth, status in zip(true_ng, statuses, strict=True)
    )
    reviewed_ng = sum(
        truth and status == "REVIEW" for truth, status in zip(true_ng, statuses, strict=True)
    )
    ng_count = sum(true_ng)
    ok_count = len(true_ng) - ng_count

    quality_indices = [
        index
        for index, (target, prediction) in enumerate(
            zip(provided_quality, quality_predictions, strict=True)
        )
        if math.isfinite(target) and math.isfinite(prediction)
    ]
    quality_f1 = None
    if quality_indices:
        threshold = (
            float(decision_cfg["quality_ng_threshold"])
            if decision_cfg.get("strategy") == "quality_head"
            else float(train_cfg.get("quality_metric_threshold", 0.5))
        )
        quality_true = [int(provided_quality[index]) for index in quality_indices]
        quality_pred = [int(quality_predictions[index] >= threshold) for index in quality_indices]
        quality_f1 = float(f1_score(quality_true, quality_pred, average="binary", zero_division=0))

    severity_pairs = [
        (target, prediction)
        for target, prediction in zip(severity_targets, severity_predictions, strict=True)
        if math.isfinite(target) and math.isfinite(prediction)
    ]
    severity_mae = (
        float(np.mean([abs(target - prediction) for target, prediction in severity_pairs]))
        if severity_pairs
        else None
    )

    return {
        "loss": float(np.mean(losses)),
        "macro_f1": macro_f1,
        "accuracy": float(np.mean(np.equal(targets, predictions))),
        "classification_report": report,
        "confusion_matrix": matrix,
        "false_accept_count": false_accept,
        "false_accept_rate": false_accept / ng_count if ng_count else None,
        "ng_recall": true_positive_ng / ng_count if ng_count else None,
        "ng_review_rate": reviewed_ng / ng_count if ng_count else None,
        "false_reject_count": false_reject,
        "false_reject_rate": false_reject / ok_count if ok_count else None,
        "ng_count": ng_count,
        "ok_count": ok_count,
        "review_count": statuses.count("REVIEW"),
        "quality_f1": quality_f1,
        "severity_mae": severity_mae,
    }


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _save_checkpoint(
    path: Path,
    *,
    model: SingleImageConvNeXt,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_selection_value: float,
    selection_metric: str,
    selection_mode: str,
    cfg: dict[str, Any],
    manifest_hash: str,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 2,
        "task_type": "single_image_classification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "best_selection_value": best_selection_value,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "class_ids": class_ids(cfg),
        "model_config": deepcopy(cfg["model"]),
        "input_config": deepcopy(cfg["input"]),
        "full_config": checkpoint_safe_config(cfg),
        "manifest_sha256": manifest_hash,
        "metrics": metrics,
        "torch_version": torch.__version__,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train(cfg: dict[str, Any]) -> Path:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    ordered_classes = class_ids(cfg)
    train_dataset, val_dataset, rows = build_datasets(cfg)
    train_cfg = cfg["train"]

    device = torch.device(choose_device(str(cfg.get("device", "auto"))))
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    model = build_model(len(ordered_classes), cfg["model"]).to(device)
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    model.set_backbone_trainable(freeze_epochs == 0)

    backbone_params = list(model.backbone.parameters()) + list(model.feature_norm.parameters())
    head_params = (
        list(model.class_head.parameters())
        + (list(model.quality_head.parameters()) if model.quality_head is not None else [])
        + (list(model.severity_head.parameters()) if model.severity_head is not None else [])
    )
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": float(train_cfg["backbone_learning_rate"])},
            {"params": head_params, "lr": float(train_cfg["learning_rate"])},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - freeze_epochs))

    class_weights = None
    if train_cfg.get("class_weighting", "auto") == "auto":
        class_weights = calculate_class_weights(rows, ordered_classes).to(device)
    class_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )

    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    output_dir = resolve_project_path(cfg, train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_project_path(cfg, train_cfg["manifest"])
    manifest_hash = sha256_file(manifest_path)
    history: list[dict[str, Any]] = []
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "macro_f1"))
    checkpoint_mode = str(train_cfg.get("checkpoint_mode", "max"))
    best_selection_value = -math.inf if checkpoint_mode == "max" else math.inf
    stale_epochs = 0
    patience = int(train_cfg.get("early_stopping_patience", 7))

    print(f"device={device}, train={len(train_dataset)}, val={len(val_dataset)}")
    for epoch in range(epochs):
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.set_backbone_trainable(True)
            print("ConvNeXt backbone fine-tuning을 시작합니다.")

        model.train()
        if epoch < freeze_epochs:
            # model.train()이 BatchNorm/Dropout 상태를 바꿀 수 있으므로 encoder는 eval로 고정합니다.
            model.backbone.eval()
            model.feature_norm.eval()
        running_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(batch["image"])
                loss, loss_parts = _compute_loss(output, batch, class_loss_fn, train_cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_losses.append(loss_parts["total"])
            progress.set_postfix(loss=f"{loss_parts['total']:.4f}")

        # frozen 구간에는 초기 LR을 유지하고, fine-tuning 시작 뒤 cosine decay를 적용합니다.
        if epoch >= freeze_epochs:
            scheduler.step()
        metrics = evaluate(
            model,
            val_loader,
            device,
            ordered_classes,
            class_loss_fn,
            train_cfg,
            cfg["decision"],
            class_display_names(cfg),
            amp_enabled,
        )
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(running_losses))
        history.append(metrics)
        print(
            f"epoch={epoch + 1} train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"false_accept={metrics['false_accept_count']}/{metrics['ng_count']}"
        )

        selection_value = metrics.get(checkpoint_metric)
        if selection_value is None or not math.isfinite(float(selection_value)):
            raise ValueError(
                f"checkpoint_metric={checkpoint_metric} 값을 계산할 수 없습니다. "
                "해당 head/라벨/threshold 설정을 확인하세요."
            )
        selection_value = float(selection_value)
        is_best = (
            selection_value > best_selection_value
            if checkpoint_mode == "max"
            else selection_value < best_selection_value
        )
        if is_best:
            best_selection_value = selection_value
            stale_epochs = 0
        else:
            stale_epochs += 1

        _save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_selection_value=best_selection_value,
            selection_metric=checkpoint_metric,
            selection_mode=checkpoint_mode,
            cfg=cfg,
            manifest_hash=manifest_hash,
            metrics=metrics,
        )
        if is_best:
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_selection_value=best_selection_value,
                selection_metric=checkpoint_metric,
                selection_mode=checkpoint_mode,
                cfg=cfg,
                manifest_hash=manifest_hash,
                metrics=metrics,
            )
        _json_dump(output_dir / "history.json", {"epochs": history})

        if stale_epochs >= patience:
            print(
                f"validation {checkpoint_metric}이(가) {patience}회 개선되지 않아 조기 종료합니다."
            )
            break

    return output_dir / "best.pt"
