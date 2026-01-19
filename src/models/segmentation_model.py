import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional
from .encoder import BaseEncoder, PlainConvEncoder
from .decoder import UNetDecoder


def init_weights_he(module, neg_slope=1e-2):
    """nnUNet-style He initialization (kaiming_normal_)"""
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=neg_slope)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        kernel_sizes: List[Union[int, Tuple[int, int]]],
        strides: List[Union[int, Tuple[int, int]]],
        conv_bias: bool = True,
        norm_op: str = "instance",
        norm_kwargs: Optional[dict] = None,
        nonlin: str = "leakyrelu",
        nonlin_kwargs: Optional[dict] = None,
        spatial_dims: int = 2,
        deep_supervision: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.spatial_dims = spatial_dims

        self.decoder = UNetDecoder(
            encoder_out_channels=encoder.out_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            kernel_sizes=kernel_sizes,
            strides=strides,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_kwargs=norm_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            spatial_dims=spatial_dims,
        )

        conv_cls = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.seg_head = conv_cls(encoder.out_channels[0], num_classes, kernel_size=1)

        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(1, len(encoder.out_channels) - 1):
                self.deep_supervision_heads.append(
                    conv_cls(encoder.out_channels[i], num_classes, kernel_size=1)
                )

        # nnUNet-style He initialization
        self.apply(init_weights_he)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        input_size = x.shape[2:]
        encoder_features = self.encoder(x)

        if self.deep_supervision and self.training:
            decoder_outputs = self.decoder(encoder_features, return_all_stages=True)
            outputs = []

            logits = self.seg_head(decoder_outputs[0])
            if logits.shape[2:] != input_size:
                logits = nn.functional.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            outputs.append(logits)

            for feat, head in zip(decoder_outputs[1:], self.deep_supervision_heads):
                outputs.append(head(feat))

            return outputs
        else:
            decoded = self.decoder(encoder_features)
            logits = self.seg_head(decoded)

            if logits.shape[2:] != input_size:
                logits = nn.functional.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

            return logits

    @classmethod
    def from_nnunet_config(
        cls,
        config: dict,
        in_channels: int = 1,
        num_classes: int = 29,
        spatial_dims: int = 2,
        deep_supervision: bool = False,
    ) -> "SegmentationModel":
        arch = config["architecture"]["arch_kwargs"]

        norm_op_str = arch.get("norm_op", "torch.nn.modules.instancenorm.InstanceNorm2d")
        norm_op = "instance" if "InstanceNorm" in norm_op_str else "batch" if "BatchNorm" in norm_op_str else "instance"
        norm_kwargs = arch.get("norm_op_kwargs", {"eps": 1e-5, "affine": True})

        nonlin_str = arch.get("nonlin", "torch.nn.LeakyReLU")
        if "LeakyReLU" in nonlin_str:
            nonlin = "leakyrelu"
        elif "ReLU" in nonlin_str:
            nonlin = "relu"
        elif "GELU" in nonlin_str:
            nonlin = "gelu"
        else:
            nonlin = "leakyrelu"
        nonlin_kwargs = arch.get("nonlin_kwargs", {"inplace": True})

        encoder = PlainConvEncoder(
            in_channels=in_channels,
            n_stages=arch["n_stages"],
            features_per_stage=arch["features_per_stage"],
            kernel_sizes=arch["kernel_sizes"],
            strides=arch["strides"],
            n_conv_per_stage=arch["n_conv_per_stage"],
            conv_bias=arch.get("conv_bias", True),
            norm_op=norm_op,
            norm_kwargs=norm_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            spatial_dims=spatial_dims,
        )

        kernel_sizes = [tuple(k) if isinstance(k, list) else k for k in arch["kernel_sizes"]]
        strides = [tuple(s) if isinstance(s, list) else s for s in arch["strides"]]

        return cls(
            encoder=encoder,
            num_classes=num_classes,
            n_conv_per_stage_decoder=arch["n_conv_per_stage_decoder"],
            kernel_sizes=kernel_sizes,
            strides=strides,
            conv_bias=arch.get("conv_bias", True),
            norm_op=norm_op,
            norm_kwargs=norm_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            spatial_dims=spatial_dims,
            deep_supervision=deep_supervision,
        )
