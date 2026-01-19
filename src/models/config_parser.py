import json
from typing import Dict, Any
from .segmentation_model import SegmentationModel


def load_nnunet_plans(plans_path: str) -> Dict[str, Any]:
    with open(plans_path, "r") as f:
        return json.load(f)


def get_configuration(plans: Dict[str, Any], config_name: str = "2d") -> Dict[str, Any]:
    if config_name not in plans["configurations"]:
        available = list(plans["configurations"].keys())
        raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")
    return plans["configurations"][config_name]


def build_model_from_plans(
    plans_path: str = None,
    plans: Dict[str, Any] = None,
    config_name: str = "2d",
    in_channels: int = 1,
    num_classes: int = 29,
    deep_supervision: bool = False,
) -> SegmentationModel:
    if plans is None:
        if plans_path is None:
            raise ValueError("Either plans_path or plans must be provided")
        plans = load_nnunet_plans(plans_path)
    config = get_configuration(plans, config_name)
    spatial_dims = 2 if "2d" in config_name.lower() else 3

    return SegmentationModel.from_nnunet_config(
        config=config,
        in_channels=in_channels,
        num_classes=num_classes,
        spatial_dims=spatial_dims,
        deep_supervision=deep_supervision,
    )


def get_decoder_config(plans_path: str, config_name: str = "2d") -> Dict[str, Any]:
    plans = load_nnunet_plans(plans_path)
    config = get_configuration(plans, config_name)
    arch = config["architecture"]["arch_kwargs"]

    norm_op_str = arch.get("norm_op", "torch.nn.modules.instancenorm.InstanceNorm2d")
    if "InstanceNorm" in norm_op_str:
        norm_op = "instance"
    elif "BatchNorm" in norm_op_str:
        norm_op = "batch"
    else:
        norm_op = "instance"

    nonlin_str = arch.get("nonlin", "torch.nn.LeakyReLU")
    if "LeakyReLU" in nonlin_str:
        nonlin = "leakyrelu"
    elif "ReLU" in nonlin_str:
        nonlin = "relu"
    elif "GELU" in nonlin_str:
        nonlin = "gelu"
    else:
        nonlin = "leakyrelu"

    spatial_dims = 2 if "2d" in config_name.lower() else 3

    return {
        "n_conv_per_stage_decoder": arch["n_conv_per_stage_decoder"],
        "kernel_sizes": [tuple(k) if isinstance(k, list) else k for k in arch["kernel_sizes"]],
        "strides": [tuple(s) if isinstance(s, list) else s for s in arch["strides"]],
        "conv_bias": arch.get("conv_bias", True),
        "norm_op": norm_op,
        "norm_kwargs": arch.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
        "nonlin": nonlin,
        "nonlin_kwargs": arch.get("nonlin_kwargs", {"inplace": True}),
        "spatial_dims": spatial_dims,
        "patch_size": config.get("patch_size"),
        "batch_size": config.get("batch_size"),
        "features_per_stage": arch["features_per_stage"],
    }


def get_normalization_params(plans_path: str, config_name: str = "2d") -> Dict[str, Any]:
    """Extract normalization parameters from plans.json (nnUNet style)"""
    plans = load_nnunet_plans(plans_path)
    config = get_configuration(plans, config_name)

    norm_schemes = config.get("normalization_schemes", ["ZScoreNormalization"])
    intensity_props = plans.get("foreground_intensity_properties_per_channel", {})

    # For single channel (grayscale)
    channel_props = intensity_props.get("0", {})

    return {
        "scheme": norm_schemes[0] if norm_schemes else "ZScoreNormalization",
        "mean": channel_props.get("mean", 0.0),
        "std": channel_props.get("std", 1.0),
        "percentile_00_5": channel_props.get("percentile_00_5", 0.0),
        "percentile_99_5": channel_props.get("percentile_99_5", 255.0),
    }
