from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from hydra.utils import get_original_cwd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_case(
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


def load_plans_patch_size(cfg) -> Tuple[int, int, int]:
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


def infer_num_classes(cfg, fallback_label: np.ndarray) -> int:
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
