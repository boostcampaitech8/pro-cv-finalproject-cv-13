from typing import Optional
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


MONAI_MODELS = {
    "attention_unet": "monai.networks.nets.AttentionUnet",
    "unet": "monai.networks.nets.UNet",
    "swin_unetr": "monai.networks.nets.SwinUNETR",
    "unetr": "monai.networks.nets.UNETR",
    "segresnet": "monai.networks.nets.SegResNet",
}


def build_model(cfg: DictConfig, num_classes: Optional[int] = None, device: str = "cuda", use_checkpoint: bool = False) -> nn.Module:
    if "_target_" in cfg:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if num_classes is not None:
            cfg_dict["out_channels"] = num_classes
        cfg_dict.pop("name", None)
        cfg_dict.pop("deep_supervision", None)
        model = instantiate(cfg_dict)
        return model.to(device)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_name = cfg_dict.pop("name")

    if use_checkpoint:
        cfg_dict["use_checkpoint"] = True

    if num_classes is not None:
        if "out_channels" in cfg_dict:
            cfg_dict["out_channels"] = num_classes
        if "num_classes" in cfg_dict:
            cfg_dict["num_classes"] = num_classes

    if model_name == "smp":
        model = _build_smp_model(cfg_dict, num_classes)

    elif model_name == "nnunet":
        from .config_parser import build_model_from_plans
        if "plans" in cfg_dict:
            plans = cfg_dict["plans"]
            plans_path = None
        else:
            plans = None
            plans_path = cfg_dict["plans_path"]
        model = build_model_from_plans(
            plans_path=plans_path,
            plans=plans,
            config_name=cfg_dict["config_name"],
            in_channels=cfg_dict["in_channels"],
            num_classes=cfg_dict["num_classes"],
            deep_supervision=cfg_dict.get("deep_supervision", False),
        )

    elif model_name in MONAI_MODELS:
        model = _build_monai_model(model_name, cfg_dict, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: smp, nnunet, {list(MONAI_MODELS.keys())}")

    return model.to(device)


def _build_smp_model(cfg_dict: dict, num_classes: Optional[int] = None) -> nn.Module:
    from .model_utils import build_smp_model, SMPDeepSupervisionWrapper

    n_classes = num_classes or cfg_dict["num_classes"]

    base_model = build_smp_model(
        encoder_name=cfg_dict["encoder_name"],
        encoder_weights=cfg_dict.get("encoder_weights"),
        in_channels=cfg_dict["in_channels"],
        num_classes=n_classes,
        decoder_type=cfg_dict["decoder_type"],
        decoder_attention_type=cfg_dict.get("decoder_attention_type"),
        norm_type=cfg_dict["norm_type"],
        num_groups=cfg_dict.get("num_groups", 32),
        encoder_depth=cfg_dict["encoder_depth"],
        decoder_channels=cfg_dict.get("decoder_channels"),
    )

    if cfg_dict.get("use_checkpoint", False):
        from torch.utils.checkpoint import checkpoint
        original_forward = base_model.encoder.forward
        def checkpointed_forward(*args, **kwargs):
            return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
        base_model.encoder.forward = checkpointed_forward

    if cfg_dict.get("deep_supervision", False):
        return SMPDeepSupervisionWrapper(base_model, n_classes)

    return base_model


def _build_monai_model(model_name: str, cfg_dict: dict, num_classes: Optional[int] = None) -> nn.Module:
    from monai.networks.nets import AttentionUnet, UNet, SwinUNETR, UNETR, SegResNet

    n_classes = num_classes or cfg_dict["out_channels"]
    in_channels = cfg_dict["in_channels"]
    spatial_dims = cfg_dict["spatial_dims"]

    if model_name == "attention_unet":
        from .model_utils import convert_norm, convert_act, MONAIDeepSupervisionWrapper

        model = AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=n_classes,
            channels=cfg_dict["channels"],
            strides=cfg_dict["strides"],
            dropout=cfg_dict.get("dropout", 0.0),
        )

        norm_type = cfg_dict["norm"]
        act_type = cfg_dict.get("act", "relu")
        if norm_type != "batch":
            model = convert_norm(model, norm_type)
        if act_type != "relu":
            model = convert_act(model, act_type)

        if cfg_dict.get("deep_supervision", False):
            model = MONAIDeepSupervisionWrapper(model, n_classes)

        return model

    elif model_name == "unet":
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=n_classes,
            channels=cfg_dict["channels"],
            strides=cfg_dict["strides"],
            dropout=cfg_dict.get("dropout", 0.0),
            norm=cfg_dict["norm"],
            act=cfg_dict.get("act", "leakyrelu"),
        )

    elif model_name == "swin_unetr":
        from .model_utils import SwinUNETRDeepSupervisionWrapper

        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=n_classes,
            feature_size=cfg_dict["feature_size"],
            spatial_dims=spatial_dims,
            depths=cfg_dict["depths"],
            num_heads=cfg_dict["num_heads"],
            norm_name=cfg_dict["norm_name"],
            drop_rate=cfg_dict.get("drop_rate", 0.0),
            use_checkpoint=cfg_dict.get("use_checkpoint", False),
        )

        if cfg_dict.get("deep_supervision", False):
            model = SwinUNETRDeepSupervisionWrapper(model, n_classes)

        return model

    elif model_name == "unetr":
        return UNETR(
            img_size=cfg_dict["img_size"],
            in_channels=in_channels,
            out_channels=n_classes,
            feature_size=cfg_dict["feature_size"],
            spatial_dims=spatial_dims,
            dropout_rate=cfg_dict.get("dropout", 0.0),
        )

    elif model_name == "segresnet":
        return SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=n_classes,
            dropout_prob=cfg_dict.get("dropout", 0.0),
        )

    raise ValueError(f"Unknown MONAI model: {model_name}")
