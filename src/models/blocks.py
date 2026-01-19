import torch
import torch.nn as nn
from typing import Tuple, Union, Optional


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        conv_bias: bool = True,
        norm_op: str = "instance",
        norm_kwargs: Optional[dict] = None,
        nonlin: str = "leakyrelu",
        nonlin_kwargs: Optional[dict] = None,
        spatial_dims: int = 2,
    ):
        super().__init__()

        if norm_kwargs is None:
            norm_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)

        conv_op = nn.Conv2d if spatial_dims == 2 else nn.Conv3d

        self.conv = conv_op(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=conv_bias,
        )
        self.norm = self._get_norm_op(norm_op, out_channels, norm_kwargs, spatial_dims)
        self.nonlin = self._get_nonlin_op(nonlin, nonlin_kwargs)

    def _get_norm_op(self, norm_op: str, num_features: int, kwargs: dict, spatial_dims: int) -> nn.Module:
        if norm_op.lower() == "instance":
            return nn.InstanceNorm2d(num_features, **kwargs) if spatial_dims == 2 else nn.InstanceNorm3d(num_features, **kwargs)
        elif norm_op.lower() == "batch":
            return nn.BatchNorm2d(num_features, **kwargs) if spatial_dims == 2 else nn.BatchNorm3d(num_features, **kwargs)
        elif norm_op.lower() == "group":
            num_groups = kwargs.pop("num_groups", 8)
            return nn.GroupNorm(num_groups, num_features, **kwargs)
        return nn.Identity()

    def _get_nonlin_op(self, nonlin: str, kwargs: dict) -> nn.Module:
        if nonlin.lower() == "leakyrelu":
            return nn.LeakyReLU(negative_slope=kwargs.pop("negative_slope", 0.01), inplace=kwargs.get("inplace", True))
        elif nonlin.lower() == "relu":
            return nn.ReLU(inplace=kwargs.get("inplace", True))
        elif nonlin.lower() == "gelu":
            return nn.GELU()
        elif nonlin.lower() in ("silu", "swish"):
            return nn.SiLU(inplace=kwargs.get("inplace", True))
        return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin(self.norm(self.conv(x)))


class StackedConvBlocks(nn.Module):
    def __init__(
        self,
        n_convs: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        first_stride: Union[int, Tuple[int, int]] = 1,
        conv_bias: bool = True,
        norm_op: str = "instance",
        norm_kwargs: Optional[dict] = None,
        nonlin: str = "leakyrelu",
        nonlin_kwargs: Optional[dict] = None,
        spatial_dims: int = 2,
    ):
        super().__init__()

        blocks = []
        for i in range(n_convs):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=first_stride if i == 0 else 1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_kwargs=norm_kwargs.copy() if norm_kwargs else None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs.copy() if nonlin_kwargs else None,
                    spatial_dims=spatial_dims,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: Union[int, Tuple[int, int]] = 2,
        mode: str = "transposed",
        spatial_dims: int = 2,
        conv_bias: bool = True,  # nnUNet default
    ):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

        if mode == "transposed":
            conv_cls = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
            self.up = conv_cls(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor, bias=conv_bias)
        else:
            conv_cls = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
            self.up = conv_cls(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transposed":
            return self.up(x)
        x = nn.functional.interpolate(
            x, scale_factor=self.scale_factor,
            mode="bilinear" if x.dim() == 4 else "trilinear", align_corners=False
        )
        return self.up(x)
