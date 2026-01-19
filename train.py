from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra.utils import get_original_cwd
from src.models.build_model import build_model as pro_build_model
from src.dataset.SegRapNPZDataset import SegRapNPZDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _split_by_case(
    files: List[Path],
    num_folds: int,
    fold: int,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    case_ids = sorted({p.parent.name for p in files})
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    num_folds = max(1, num_folds)
    fold = max(0, min(fold, num_folds - 1))
    fold_size = math.ceil(len(case_ids) / num_folds)
    start = fold * fold_size
    end = min(len(case_ids), start + fold_size)
    val_cases = set(case_ids[start:end])

    train_files = [p for p in files if p.parent.name not in val_cases]
    val_files = [p for p in files if p.parent.name in val_cases]
    return train_files, val_files


def build_model(cfg: DictConfig, in_channels: int, num_classes: int, device: torch.device) -> nn.Module:
    with open_dict(cfg.model):
        cfg.model.in_channels = in_channels
        cfg.model.num_classes = num_classes

    return pro_build_model(
        cfg["model"],
        num_classes=num_classes,
        device=str(device),
        use_checkpoint=bool(cfg["model"]["use_checkpoint"]),
    )


def _infer_num_classes(cfg: DictConfig, fallback_label: np.ndarray) -> int:
    model_cfg = cfg["model"]
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])


    dataset_json = cfg["data"]["dataset_json_template"]
    path = Path(dataset_json)
    if not path.is_absolute():
        path = Path(get_original_cwd()) / path
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        labels = dataset.get("labels", {})
        if "background" in labels:
            return max(int(v) for v in labels.values()) + 1
        if all(str(k).isdigit() for k in labels.keys()):
            return max(int(k) for k in labels.keys()) + 1

    return int(fallback_label.max()) + 1


def _dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(target, (list, tuple)):
        target = target[0]
    pred = pred.argmax(1)
    dice_scores = []
    for cls in range(1, num_classes):
        pred_c = pred == cls
        target_c = target == cls
        inter = (pred_c & target_c).sum().item()
        denom = pred_c.sum().item() + target_c.sum().item()
        if denom == 0:
            continue
        dice_scores.append(2.0 * inter / (denom + 1e-6))
    if not dice_scores:
        return 0.0
    return float(sum(dice_scores) / len(dice_scores))


def _resize_target(target: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    resized = torch.nn.functional.interpolate(target[:, None].float(), size=shape, mode="nearest")
    return resized[:, 0].long()


def _prepare_targets(outputs, target):
    if not isinstance(outputs, (list, tuple)):
        return target
    targets = []
    for out in outputs:
        if out.shape[2:] == target.shape[1:]:
            targets.append(target)
        else:
            targets.append(_resize_target(target, out.shape[2:]))
    return targets


def _compute_loss(loss_fn, outputs, targets) -> torch.Tensor:
    if not isinstance(outputs, (list, tuple)):
        return loss_fn(outputs, targets)
    weights = [1.0 / len(outputs)] * len(outputs)
    loss = 0.0
    for out, tgt, w in zip(outputs, targets, weights):
        loss = loss + w * loss_fn(out, tgt)
    return loss


def _save_checkpoint(output_dir: Path, epoch: int, model: nn.Module, optimizer, best_dice: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "checkpoint_best.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
        },
        save_path,
    )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    preprocessed_dir = Path(cfg["data"]["preprocessed_dir"])
    if not preprocessed_dir.is_absolute():
        preprocessed_dir = Path(get_original_cwd()) / preprocessed_dir
    files = sorted(preprocessed_dir.glob("*/*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {preprocessed_dir}")

    patch_size = tuple(cfg["train"]["patch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    train_files, val_files = _split_by_case(
        files,
        num_folds=int(cfg.train.num_folds),
        fold=int(cfg.train.fold),
        seed=int(cfg.seed),
    )

    sample = np.load(train_files[0])
    in_channels = int(sample["image"].shape[0])
    num_classes = _infer_num_classes(cfg, sample["label"])
    sample.close()

    train_dataset = SegRapNPZDataset(train_files, patch_size=patch_size, is_train=True)
    val_dataset = SegRapNPZDataset(val_files, patch_size=patch_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(cfg, in_channels=in_channels, num_classes=num_classes, device=device)

    if cfg["model"]["compile"] and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if cfg.train.use_amp and device.type == "cuda" else None

    best_dice = -1.0
    output_dir = Path(cfg.train.output_dir)

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = infinite(train_loader)
    val_iter = infinite(val_loader)

    for epoch in range(cfg.train.epochs):
        model.train()
        train_losses = []
        for _ in tqdm(
            range(cfg.train.num_iterations_per_epoch),
            desc=f"Epoch {epoch + 1} train",
            unit="batch",
        ):
            data, target = next(train_iter)
            data = data.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type=device.type, enabled=True):
                    logits = model(data)
                    prepared_target = _prepare_targets(logits, target)
                    loss = _compute_loss(loss_fn, logits, prepared_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(data)
                prepared_target = _prepare_targets(logits, target)
                loss = _compute_loss(loss_fn, logits, prepared_target)
                loss.backward()
                optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_dices = []
        with torch.no_grad():
            for _ in tqdm(
                range(cfg.train.num_val_iterations),
                desc=f"Epoch {epoch + 1} val",
                unit="batch",
            ):
                data, target = next(val_iter)
                data = data.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.long)
                logits = model(data)
                prepared_target = _prepare_targets(logits, target)
                loss = _compute_loss(loss_fn, logits, prepared_target)
                val_losses.append(float(loss.detach().cpu()))
                val_dices.append(_dice_score(logits, prepared_target, num_classes))

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        mean_val_dice = float(np.mean(val_dices))

        print(
            f"Epoch {epoch + 1}/{cfg.train.epochs} | "
            f"train_loss: {mean_train_loss:.4f} | "
            f"val_loss: {mean_val_loss:.4f} | "
            f"val_dice: {mean_val_dice:.4f}"
        )

        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            _save_checkpoint(output_dir, epoch, model, optimizer, best_dice)

    print(f"Training done. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()
