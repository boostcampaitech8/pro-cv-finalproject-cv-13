import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
from torch.utils.checkpoint import checkpoint
from .blocks import StackedConvBlocks


class BaseEncoder(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def num_stages(self) -> int:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass


class PlainConvEncoder(BaseEncoder):
    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        kernel_sizes: List[Union[int, Tuple[int, int]]],
        strides: List[Union[int, Tuple[int, int]]],
        n_conv_per_stage: List[int],
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

        self.n_stages = n_stages
        self._out_channels = features_per_stage
        self._strides = strides
        self.spatial_dims = spatial_dims
        self.grad_checkpointing = False

        self.stages = nn.ModuleList()

        for stage_idx in range(n_stages):
            stage_in_channels = in_channels if stage_idx == 0 else features_per_stage[stage_idx - 1]
            stage_out_channels = features_per_stage[stage_idx]

            stride = strides[stage_idx]
            if isinstance(stride, int):
                stride = (stride,) * spatial_dims

            stage = StackedConvBlocks(
                n_convs=n_conv_per_stage[stage_idx],
                in_channels=stage_in_channels,
                out_channels=stage_out_channels,
                kernel_size=kernel_sizes[stage_idx],
                first_stride=stride,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_kwargs=norm_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                spatial_dims=spatial_dims,
            )
            self.stages.append(stage)

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    @property
    def num_stages(self) -> int:
        return self.n_stages

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            if self.grad_checkpointing and self.training:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            features.append(x)
        return features


