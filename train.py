from __future__ import annotations

import json
import inspect
import math
import random
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    with open_dict(cfg["model"]):
        cfg["model"]["in_channels"] = in_channels
        cfg["model"]["num_classes"] = num_classes

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


    dataset_json = cfg["dataset"]["dataset_json_template"]
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


def _load_plans_patch_size(cfg: DictConfig) -> Tuple[int, int, int]:
    plans_path = Path(cfg["model"]["plans_path"])
    if not plans_path.is_absolute():
        plans_path = Path(get_original_cwd()) / plans_path
    with plans_path.open("r", encoding="utf-8") as f:
        plans = json.load(f)

    config_name = cfg["model"]["config_name"]
    config = plans["configurations"][config_name]
    patch_size = config.get("patch_size")
    if patch_size is None:
        raise KeyError(f"No patch_size found in plans for config {config_name}")
    return tuple(patch_size)


def _get_sliding_window_starts(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _sliding_window_inference(
    volume: torch.Tensor,
    model: nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> torch.Tensor:
    _, d, h, w = volume.shape
    pd = max(0, patch_size[0] - d)
    ph = max(0, patch_size[1] - h)
    pw = max(0, patch_size[2] - w)

    if pd or ph or pw:
        volume = torch.nn.functional.pad(volume, (0, pw, 0, ph, 0, pd))

    _, d_p, h_p, w_p = volume.shape
    stride_d = max(1, int(patch_size[0] * (1 - overlap)))
    stride_h = max(1, int(patch_size[1] * (1 - overlap)))
    stride_w = max(1, int(patch_size[2] * (1 - overlap)))

    starts_d = _get_sliding_window_starts(d_p, patch_size[0], stride_d)
    starts_h = _get_sliding_window_starts(h_p, patch_size[1], stride_h)
    starts_w = _get_sliding_window_starts(w_p, patch_size[2], stride_w)

    output_sum = None
    count = torch.zeros((1, d_p, h_p, w_p), dtype=torch.float32)

    for sd in starts_d:
        for sh in starts_h:
            for sw in starts_w:
                patch = volume[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ]
                patch = patch.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(patch)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                logits = logits.detach().cpu()[0]

                if output_sum is None:
                    output_sum = torch.zeros(
                        (logits.shape[0], d_p, h_p, w_p),
                        dtype=torch.float32,
                    )
                output_sum[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ] += logits
                count[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ] += 1

    output_sum = output_sum / count
    return output_sum[:, :d, :h, :w]


def _compute_loss(loss_fn, outputs, targets) -> torch.Tensor:
    def _match_target_shape(output_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        if target_tensor.ndim == output_tensor.ndim - 1:
            return target_tensor.unsqueeze(1)
        return target_tensor

    if not isinstance(outputs, (list, tuple)):
        targets = _match_target_shape(outputs, targets)
        return loss_fn(outputs, targets)
    weights = [1.0 / len(outputs)] * len(outputs)
    loss = 0.0
    for out, tgt, w in zip(outputs, targets, weights):
        tgt = _match_target_shape(out, tgt)
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
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    preprocessed_dir = Path(cfg["dataset"]["preprocessed_dir"])
    if not preprocessed_dir.is_absolute():
        preprocessed_dir = Path(get_original_cwd()) / preprocessed_dir
    files = sorted(preprocessed_dir.glob("*/*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {preprocessed_dir}")

    if "patch_size" in cfg["train"] and cfg["train"]["patch_size"] is not None:
        patch_size = tuple(cfg["train"]["patch_size"])
    else:
        patch_size = _load_plans_patch_size(cfg)
    num_workers = int(cfg["train"]["num_workers"])
    val_sliding_window = bool(cfg["train"]["val_sliding_window"])
    val_overlap = float(cfg["train"]["val_overlap"])

    train_files, val_files = _split_by_case(
        files,
        num_folds=int(cfg["train"]["num_folds"]),
        fold=int(cfg["train"]["fold"]),
        seed=int(cfg["seed"]),
    )

    sample = np.load(train_files[0])
    in_channels = int(sample["image"].shape[0])
    num_classes = _infer_num_classes(cfg, sample["label"])
    sample.close()

    train_batch_size = int(cfg["train"]["batch_size"])
    val_batch_size = int(cfg["train"].get("val_batch_size", train_batch_size))

    train_dataset = SegRapNPZDataset(
        train_files,
        patch_size=patch_size,
        is_train=True,
        oversample_foreground_percent=float(cfg["train"]["oversample_foreground_percent"]),
    )
    val_dataset = SegRapNPZDataset(val_files, patch_size=patch_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if not val_sliding_window:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    else:
        val_loader = None

    model = build_model(cfg, in_channels=in_channels, num_classes=num_classes, device=device)

    if cfg["model"]["compile"] and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer_cfg = OmegaConf.to_container(cfg["optimizer"], resolve=True)
    optimizer_name = optimizer_cfg.pop("name")
    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_cls(model.parameters(), **optimizer_cfg)

    loss_cfg = OmegaConf.to_container(cfg["loss"], resolve=True)
    loss_cfg.pop("name", None)
    if "_target_" in loss_cfg:
        loss_target = loss_cfg["_target_"]
        loss_kwargs = {k: v for k, v in loss_cfg.items() if k != "_target_"}
        loss_cls = hydra.utils.get_class(loss_target)
        sig = inspect.signature(loss_cls.__init__)
        filtered_kwargs = {k: v for k, v in loss_kwargs.items() if k in sig.parameters}
        loss_fn = loss_cls(**filtered_kwargs)
    else:
        loss_name = loss_cfg["name"]
        raise ValueError(f"Unsupported loss config: {loss_name}")
    
    scheduler_cfg = cfg["scheduler"]
    if scheduler_cfg["name"] == "poly":
        max_epochs = int(cfg["train"]["epochs"])
        exponent = float(scheduler_cfg["exponent"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: (1 - e / max_epochs) ** exponent
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg['name']}")

    scaler = torch.amp.GradScaler("cuda") if cfg["train"]["use_amp"] and device.type == "cuda" else None

    best_dice = -1.0
    output_dir = Path(cfg["train"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(get_original_cwd()) / output_dir

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = infinite(train_loader)
    val_iter = infinite(val_loader) if val_loader is not None else None

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_losses = []
        for _ in tqdm(
            range(cfg["train"]["num_iterations_per_epoch"]),
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
        if val_sliding_window:
            for val_path in tqdm(val_files, desc=f"Epoch {epoch + 1} val", unit="case"):
                with np.load(val_path) as data:
                    image = torch.from_numpy(data["image"].astype(np.float32))
                    target = torch.from_numpy(data["label"].astype(np.int64))
                logits = _sliding_window_inference(
                    image, model, patch_size=patch_size, overlap=val_overlap, device=device
                )
                loss = _compute_loss(loss_fn, logits.unsqueeze(0), target.unsqueeze(0))
                val_losses.append(float(loss.detach().cpu()))
                val_dices.append(_dice_score(logits.unsqueeze(0), target.unsqueeze(0), num_classes))
        else:
            with torch.no_grad():
                for _ in tqdm(
                    range(cfg["train"]["num_val_iterations"]),
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

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg['train']['epochs']} | "
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
