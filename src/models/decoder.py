import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional
from .blocks import StackedConvBlocks, UpsampleBlock


class UNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_out_channels: List[int],
        n_conv_per_stage_decoder: List[int],
        kernel_sizes: List[Union[int, Tuple[int, int]]],
        strides: List[Union[int, Tuple[int, int]]],
        conv_bias: bool = True,
        norm_op: str = "instance",
        norm_kwargs: Optional[dict] = None,
        nonlin: str = "leakyrelu",
        nonlin_kwargs: Optional[dict] = None,
        spatial_dims: int = 2,
        upsample_mode: str = "transposed",
    ):
        super().__init__()

        if norm_kwargs is None:
            norm_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.spatial_dims = spatial_dims
        n_stages_encoder = len(encoder_out_channels)
        n_stages_decoder = n_stages_encoder - 1

        self.upsample_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for stage_idx in range(n_stages_decoder):
            encoder_stage_idx = n_stages_encoder - 2 - stage_idx

            in_channels_upsample = encoder_out_channels[encoder_stage_idx + 1]
            out_channels_upsample = encoder_out_channels[encoder_stage_idx]

            upsample_stride = strides[encoder_stage_idx + 1]
            if isinstance(upsample_stride, int):
                upsample_stride = (upsample_stride,) * spatial_dims

            self.upsample_blocks.append(
                UpsampleBlock(
                    in_channels=in_channels_upsample,
                    out_channels=out_channels_upsample,
                    scale_factor=upsample_stride,
                    mode=upsample_mode,
                    spatial_dims=spatial_dims,
                    conv_bias=conv_bias,  # nnUNet style
                )
            )

            conv_in_channels = encoder_out_channels[encoder_stage_idx] * 2
            conv_out_channels = encoder_out_channels[encoder_stage_idx]

            self.conv_blocks.append(
                StackedConvBlocks(
                    n_convs=n_conv_per_stage_decoder[stage_idx],
                    in_channels=conv_in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=kernel_sizes[encoder_stage_idx],
                    first_stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_kwargs=norm_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    spatial_dims=spatial_dims,
                )
            )

    def forward(
        self, encoder_features: List[torch.Tensor], return_all_stages: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = encoder_features[-1]
        decoder_outputs = []

        for stage_idx, (upsample, conv) in enumerate(zip(self.upsample_blocks, self.conv_blocks)):
            x = upsample(x)

            skip_idx = len(encoder_features) - 2 - stage_idx
            skip = encoder_features[skip_idx]

            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear" if self.spatial_dims == 2 else "trilinear"
                )

            x = torch.cat([skip, x], dim=1)
            x = conv(x)

            if return_all_stages:
                decoder_outputs.append(x)

        if return_all_stages:
            return decoder_outputs[::-1]
        return x
