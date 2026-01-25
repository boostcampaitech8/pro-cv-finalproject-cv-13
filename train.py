from __future__ import annotations

import inspect
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augment.nnunet_augmentation import AugmentConfig, NnUNetAugment3D
from src.dataset.SegRapNPZDataset import SegRapNPZDataset, SegRapRAMDataset
from src.models.build_model import build_model as pro_build_model
from src.training.inference import sliding_window_inference
from src.training.metrics import dice_score, pick_highest_res_output
from src.training.targets import prepare_targets
from src.training.utils import infer_num_classes, load_plans_patch_size, set_seed, split_by_case


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


def _save_checkpoint(output_dir: Path, epoch: int, model: nn.Module, optimizer, best_dice: float) -> None:
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
def train(cfg: DictConfig) -> None:
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    use_ram_cache = bool(cfg["dataset"].get("use_ram_cache", False))
    patch_size = tuple(cfg["train"]["patch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    val_sliding_window = bool(cfg["train"]["val_sliding_window"])
    val_overlap = float(cfg["train"]["val_overlap"])

    if use_ram_cache:
        raw_dir = Path(cfg["dataset"]["raw_dir"])
        labels_dir = Path(cfg["dataset"]["labels_dir"])
        if not raw_dir.is_absolute():
            raw_dir = Path(get_original_cwd()) / raw_dir
        if not labels_dir.is_absolute():
            labels_dir = Path(get_original_cwd()) / labels_dir

        modalities = list(cfg["dataset"].get("modalities", ["image.nii.gz", "image_contrast.nii.gz"]))
        case_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
        if not case_dirs:
            raise FileNotFoundError(f"No case directories found under {raw_dir}")

        case_paths = []
        for case_dir in case_dirs:
            img_path = case_dir / modalities[0]
            if not img_path.exists():
                raise FileNotFoundError(f"Missing modality file: {img_path}")
            case_paths.append(img_path)

        train_files, val_files = split_by_case(
            case_paths,
            num_folds=int(cfg["train"]["num_folds"]),
            fold=int(cfg["train"]["fold"]),
            seed=int(cfg["seed"]),
        )
        train_cases = [p.parent for p in train_files]
        val_cases = [p.parent for p in val_files]

        cache_in_ram = bool(cfg["dataset"].get("cache_in_ram", True))
        cached_cases = (
            SegRapRAMDataset.cache_cases(case_dirs, labels_dir, modalities) if cache_in_ram else None
        )

        train_dataset = SegRapRAMDataset(
            train_cases,
            labels_dir=labels_dir,
            modalities=modalities,
            patch_size=patch_size,
            is_train=True,
            oversample_foreground_percent=float(cfg["train"]["oversample_foreground_percent"]),
            cache_in_ram=cache_in_ram,
            cached_cases=cached_cases,
        )
        val_dataset = SegRapRAMDataset(
            val_cases,
            labels_dir=labels_dir,
            modalities=modalities,
            patch_size=patch_size,
            is_train=False,
            cache_in_ram=cache_in_ram,
            cached_cases=cached_cases,
        )

        sample_image, sample_label = train_dataset.get_full_case(0)
        in_channels = int(sample_image.shape[0])
        num_classes = infer_num_classes(cfg, sample_label)
    else:
        preprocessed_dir = Path(cfg["dataset"]["preprocessed_dir"])
        if not preprocessed_dir.is_absolute():
            preprocessed_dir = Path(get_original_cwd()) / preprocessed_dir
        files = sorted(preprocessed_dir.glob("*/*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found under {preprocessed_dir}")

        train_files, val_files = split_by_case(
            files,
            num_folds=int(cfg["train"]["num_folds"]),
            fold=int(cfg["train"]["fold"]),
            seed=int(cfg["seed"]),
        )

        sample = np.load(train_files[0])
        in_channels = int(sample["image"].shape[0])
        num_classes = infer_num_classes(cfg, sample["label"])
        sample.close()

    train_batch_size = int(cfg["train"]["batch_size"])
    val_batch_size = int(cfg["train"].get("val_batch_size", train_batch_size))

    if not use_ram_cache:
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

    augmenter = None
    if bool(cfg["train"].get("use_augmentation", False)):
        aug_cfg = cfg.get("augmentation", {})
        aug_kwargs = {k: v for k, v in aug_cfg.items() if k in AugmentConfig.__annotations__}
        augmenter = NnUNetAugment3D(AugmentConfig(**aug_kwargs))

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
            if augmenter is not None:
                with torch.no_grad():
                    data, target = augmenter(data, target)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type=device.type, enabled=True):
                    logits = model(data)
                    prepared_target = prepare_targets(logits, target)
                    loss = _compute_loss(loss_fn, logits, prepared_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(data)
                prepared_target = prepare_targets(logits, target)
                loss = _compute_loss(loss_fn, logits, prepared_target)
                loss.backward()
                optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_dices = []
        if val_sliding_window:
            if use_ram_cache:
                for idx in tqdm(range(len(val_dataset)), desc=f"Epoch {epoch + 1} val", unit="case"):
                    image_np, target_np = val_dataset.get_full_case(idx)
                    image = torch.from_numpy(np.ascontiguousarray(image_np)).float()
                    target = torch.from_numpy(np.ascontiguousarray(target_np)).long()
                    logits = sliding_window_inference(
                        image, model, patch_size=patch_size, overlap=val_overlap, device=device
                    )
                    loss = _compute_loss(loss_fn, logits.unsqueeze(0), target.unsqueeze(0))
                    val_losses.append(float(loss.detach().cpu()))
                    val_dices.append(dice_score(logits.unsqueeze(0), target.unsqueeze(0), num_classes))
            else:
                for val_path in tqdm(val_files, desc=f"Epoch {epoch + 1} val", unit="case"):
                    with np.load(val_path) as data:
                        image = torch.from_numpy(data["image"].astype(np.float32))
                        target = torch.from_numpy(data["label"].astype(np.int64))
                    logits = sliding_window_inference(
                        image, model, patch_size=patch_size, overlap=val_overlap, device=device
                    )
                    loss = _compute_loss(loss_fn, logits.unsqueeze(0), target.unsqueeze(0))
                    val_losses.append(float(loss.detach().cpu()))
                    val_dices.append(dice_score(logits.unsqueeze(0), target.unsqueeze(0), num_classes))
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
                    prepared_target = prepare_targets(logits, target)
                    loss = _compute_loss(loss_fn, logits, prepared_target)
                    val_losses.append(float(loss.detach().cpu()))
                    dice_logits, dice_target = pick_highest_res_output(logits, prepared_target)
                    val_dices.append(dice_score(dice_logits, dice_target, num_classes))

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
