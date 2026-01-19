from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def _dataset_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def load_plans_and_dataset(preprocessed_base: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plans = load_json(str(preprocessed_base / "nnUNetPlans.json"))
    dataset_json = load_json(str(preprocessed_base / "dataset.json"))
    return plans, dataset_json


def _load_split(preprocessed_base: Path, fold: int, n_splits: int, seed: int, identifiers):
    splits_file = preprocessed_base / "splits_final.json"
    if splits_file.exists():
        splits = load_json(str(splits_file))
        if fold < 0 or fold >= len(splits):
            raise ValueError(f"Fold {fold} is out of range for splits_final.json")
        return splits[fold]["train"], splits[fold]["val"]

    splits = generate_crossval_split(identifiers, seed=seed, n_splits=n_splits)
    return splits[fold]["train"], splits[fold]["val"]


def build_nnunet_dataloaders(cfg):
    dataset_name = _dataset_name(cfg.data.dataset_id, cfg.data.dataset_name)
    preprocessed_base = Path(cfg.data.nnunet_preprocessed) / dataset_name

    plans, dataset_json = load_plans_and_dataset(preprocessed_base)
    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration(cfg.model.configuration)

    data_identifier = config_manager.data_identifier
    dataset_folder = preprocessed_base / data_identifier

    dataset_class = infer_dataset_class(str(dataset_folder))
    base_dataset = dataset_class(str(dataset_folder))
    identifiers = base_dataset.identifiers

    train_ids, val_ids = _load_split(
        preprocessed_base,
        fold=cfg.train.fold,
        n_splits=cfg.train.num_folds,
        seed=cfg.seed,
        identifiers=identifiers,
    )

    train_dataset = dataset_class(str(dataset_folder), identifiers=train_ids)
    val_dataset = dataset_class(str(dataset_folder), identifiers=val_ids)

    label_manager = plans_manager.get_label_manager(dataset_json)
    patch_size = config_manager.patch_size

    train_loader = nnUNetDataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        patch_size=patch_size,
        final_patch_size=patch_size,
        label_manager=label_manager,
        oversample_foreground_percent=cfg.train.oversample_foreground_percent,
        probabilistic_oversampling=False,
        transforms=None,
    )

    val_loader = nnUNetDataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        patch_size=patch_size,
        final_patch_size=patch_size,
        label_manager=label_manager,
        oversample_foreground_percent=0.0,
        probabilistic_oversampling=False,
        transforms=None,
    )

    return train_loader, val_loader, plans_manager, config_manager, label_manager, dataset_json
