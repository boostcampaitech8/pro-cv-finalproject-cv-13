import torch
import torch.nn as nn
from typing import Tuple, Sequence
from monai.networks.nets.attentionunet import AttentionUnet, AttentionLayer, ConvBlock
from monai.networks.blocks import Convolution


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module"""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        norm: str,
        atrous_rates: Tuple[int, ...] = (6, 12, 18),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims

        self.conv1x1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm,
            act="relu",
            dropout=dropout,
        )

        self.atrous_convs = nn.ModuleList([
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                norm=norm,
                act="relu",
                dropout=dropout,
            )
            for rate in atrous_rates
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)
        # conv_only=True to avoid BatchNorm on 1x1 spatial size
        self.pool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1) if spatial_dims == 2 else nn.Conv3d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
        )

        total_channels = out_channels * (len(atrous_rates) + 2)
        self.project = Convolution(
            spatial_dims=spatial_dims,
            in_channels=total_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm,
            act="relu",
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]

        features = [self.conv1x1(x)]
        features.extend(conv(x) for conv in self.atrous_convs)

        pool_feat = self.pool_conv(self.global_pool(x))
        mode = 'bilinear' if self.spatial_dims == 2 else 'trilinear'
        pool_feat = nn.functional.interpolate(pool_feat, size=size, mode=mode, align_corners=False)
        features.append(pool_feat)

        return self.project(torch.cat(features, dim=1))


class AttentionUnetASPP(AttentionUnet):
    """
    Attention U-Net with ASPP in the bottleneck.

    Inherits from MONAI's AttentionUnet and overrides _get_bottom_layer
    to add ASPP module after the bottleneck convolution.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        norm: str,
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        dropout: float = 0.0,
        aspp_rates: Tuple[int, ...] = (6, 12, 18),
    ):
        self.aspp_rates = aspp_rates
        self.norm = norm
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            dropout=dropout,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        """Override to add ASPP after bottleneck convolution"""
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=nn.Sequential(
                ConvBlock(
                    spatial_dims=self.dimensions,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    dropout=self.dropout,
                    kernel_size=self.kernel_size,
                ),
                ASPP(
                    spatial_dims=self.dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm=self.norm,
                    atrous_rates=self.aspp_rates,
                    dropout=self.dropout,
                ),
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )
