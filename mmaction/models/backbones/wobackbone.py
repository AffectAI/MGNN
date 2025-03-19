import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class WOBackbone(BaseModule):

    def __init__(self):
        pass
        super().__init__()

    def forward(self, x):
        # shape: [B*H, C, H, W]
        outs = []
        x = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        outs.append(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        outs.append(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        outs.append(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        outs.append(x)

        return tuple(outs)