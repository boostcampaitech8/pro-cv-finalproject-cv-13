import torch
import torch.nn as nn
from typing import Optional

import segmentation_models_pytorch as smp


DECODER_MAP = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "pspnet": smp.PSPNet,
}


NORM_TYPES = (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)


def get_num_channels(norm_layer: nn.Module) -> int:
    """Normalization 레이어에서 채널 수 추출"""
    if isinstance(norm_layer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        return norm_layer.num_features
    elif isinstance(norm_layer, nn.GroupNorm):
        return norm_layer.num_channels
    return 0


def create_norm_layer(norm_type: str, num_channels: int, num_groups: int = 32) -> nn.Module:
    """지정된 타입의 normalization 레이어 생성"""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels, affine=True)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm_type == "group":
        # num_channels가 num_groups로 나눠지지 않으면 조정
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups, num_channels, affine=True)
    elif norm_type == "layer":
        # LayerNorm = GroupNorm with num_groups=1
        return nn.GroupNorm(1, num_channels, affine=True)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Choose from: batch, instance, group, layer")


def convert_norm(module: nn.Module, norm_type: str = "batch", num_groups: int = 32) -> nn.Module:
    """모든 normalization 레이어를 지정된 타입으로 변환 (재귀적)"""
    for name, child in module.named_children():
        if isinstance(child, NORM_TYPES):
            num_channels = get_num_channels(child)
            new_norm = create_norm_layer(norm_type, num_channels, num_groups)
            setattr(module, name, new_norm)
        else:
            convert_norm(child, norm_type, num_groups)
    return module


def convert_act(module: nn.Module, act_type: str = "leakyrelu") -> nn.Module:
    """모든 activation 레이어를 지정된 타입으로 변환 (재귀적)"""
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.PReLU)):
            if act_type == "leakyrelu":
                setattr(module, name, nn.LeakyReLU(negative_slope=0.01, inplace=True))
            elif act_type == "relu":
                setattr(module, name, nn.ReLU(inplace=True))
            elif act_type == "prelu":
                setattr(module, name, nn.PReLU())
        else:
            convert_act(child, act_type)
    return module


def build_smp_model(
    encoder_name: str = "efficientnet-b1",
    encoder_weights: Optional[str] = None,
    in_channels: int = 1,
    num_classes: int = 30,
    decoder_type: str = "unet",
    decoder_attention_type: Optional[str] = None,  # "scse" for Attention U-Net
    norm_type: str = "batch",  # batch, instance, group, layer
    num_groups: int = 32,  # GroupNorm용
    encoder_depth: int = 5,  # encoder 깊이 (3-5)
    decoder_channels: Optional[tuple] = None,  # decoder 채널 수
) -> nn.Module:
    if decoder_type not in DECODER_MAP:
        raise ValueError(f"Unknown decoder: {decoder_type}. Choose from {list(DECODER_MAP.keys())}")

    model_cls = DECODER_MAP[decoder_type]
   
    model_kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": num_classes,
        "encoder_depth": encoder_depth,
    }
    if decoder_attention_type is not None:
        model_kwargs["decoder_attention_type"] = decoder_attention_type
    if decoder_channels is not None:
        model_kwargs["decoder_channels"] = decoder_channels

    model = model_cls(**model_kwargs)

    if norm_type != "batch":
        model = convert_norm(model, norm_type, num_groups)

    return model


class BaseDeepSupervisionWrapper(nn.Module):
    """Deep supervision wrapper 베이스 클래스"""
    reverse_features = False  

    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.intermediate_features = []
        self.hooks = []
        self.num_ds_outputs = 0
       
        decoder_channels = self._find_decoder_layers()

        if decoder_channels:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(ch, num_classes, kernel_size=1)
                for ch in decoder_channels
            ])
            self.num_ds_outputs = len(decoder_channels)
        else:
            self.ds_heads = None

    def _find_decoder_layers(self) -> list:       
        raise NotImplementedError

    def _get_hook(self):
        def hook(module, input, output):
            if self.training:
                self.intermediate_features.append(output)
        return hook

    def forward(self, x):
        self.intermediate_features = []

        if not self.training or self.ds_heads is None:
            return self.model(x)

        final_output = self.model(x)
        outputs = [final_output]

        features = self.intermediate_features[-self.num_ds_outputs:] if self.num_ds_outputs > 0 else []
        if self.reverse_features:
            features = list(reversed(features))

        for i, feat in enumerate(features):
            ds_output = self.ds_heads[i](feat)
            outputs.append(ds_output)

        return outputs

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()


class MONAIDeepSupervisionWrapper(BaseDeepSupervisionWrapper):
    """MONAI 모델 (AttentionUnet, UNet 등)에 deep supervision 추가"""
    reverse_features = True  # MONAI는 deep→shallow 순서로 수집되어 reverse 필요

    def _find_decoder_layers(self) -> list:
        from monai.networks.nets.attentionunet import AttentionLayer

        channels = []
        for _, child in self.model.named_modules():
            if isinstance(child, AttentionLayer):
                if hasattr(child, 'upconv') and hasattr(child.upconv, 'up'):
                    out_ch = child.upconv.up.conv.out_channels
                    channels.append(out_ch)
                    hook = child.upconv.register_forward_hook(self._get_hook())
                    self.hooks.append(hook)

        # 고해상도 출력만 사용 (최대 4개)
        max_ds_layers = min(4, len(channels))
        return channels[:max_ds_layers]


class SwinUNETRDeepSupervisionWrapper(BaseDeepSupervisionWrapper):
    """SwinUNETR에 deep supervision 추가"""

    def _find_decoder_layers(self) -> list:
        decoder_names = ['decoder4', 'decoder3', 'decoder2', 'decoder1']
        channels = []

        for name in decoder_names:
            if hasattr(self.model, name):
                decoder = getattr(self.model, name)
                if hasattr(decoder, 'conv_block'):
                    out_ch = decoder.conv_block.conv2.conv.out_channels
                    channels.append(out_ch)
                    hook = decoder.register_forward_hook(self._get_hook())
                    self.hooks.append(hook)

        return channels


class SMPDeepSupervisionWrapper(BaseDeepSupervisionWrapper):
    """SMP 모델에 deep supervision 추가"""

    def _find_decoder_layers(self) -> list:
        channels = []

        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'blocks'):
            for i, block in enumerate(self.model.decoder.blocks[:-1]):
                if hasattr(block, 'conv2'):
                    out_ch = block.conv2[0].out_channels
                else:
                    out_ch = 256 // (2 ** i)
                channels.append(out_ch)
                hook = block.register_forward_hook(self._get_hook())
                self.hooks.append(hook)

        return channels
