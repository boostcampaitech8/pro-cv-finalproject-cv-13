import torch
import torch.nn as nn


class BoundaryWrapper(nn.Module):
    """
    Wrapper that adds boundary prediction head to existing segmentation model.
    Uses forward hook to capture decoder features without modifying base model.
    """

    def __init__(self, base_model, feature_channels=None):
        super().__init__()
        self.base_model = base_model
        self._features = None
        self._hook = None

        # feature_channels 자동 추론
        if feature_channels is None:
            if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'out_channels'):
                feature_channels = base_model.encoder.out_channels[0]
            else:
                raise ValueError("Cannot infer feature_channels. Please provide explicitly.")

        # 디코더 마지막 conv_block (가장 높은 해상도)에 hook 등록
        if hasattr(base_model, 'decoder') and hasattr(base_model.decoder, 'conv_blocks'):
            target_layer = base_model.decoder.conv_blocks[-1]
            self._hook = target_layer.register_forward_hook(self._hook_fn)
        else:
            raise ValueError("base_model must have decoder.conv_blocks")

        self.boundary_head = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feature_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, 1, kernel_size=1),
        )

    def _hook_fn(self, module, input, output):
        self._features = output

    def forward(self, x):
        self._features = None

        seg_pred = self.base_model(x)

        if self._features is None:
            raise RuntimeError("Hook failed: decoder features not captured. Check model structure.")

        boundary_pred = self.boundary_head(self._features)

        if self.training:
            return seg_pred, boundary_pred
        else:
            return seg_pred

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
