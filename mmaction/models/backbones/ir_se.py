from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter, ModuleList, SyncBatchNorm
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
from typing import Dict, List, Optional, Sequence, Tuple, Union
import mmengine
from mmengine.logging import MMLogger
from mmaction.registry import MODELS
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import _load_checkpoint
from mmaction.utils import ConfigType


##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), SyncBatchNorm(depth))
        self.res_layer = Sequential(
            SyncBatchNorm(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), SyncBatchNorm(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                SyncBatchNorm(depth))
        self.res_layer = Sequential(
            SyncBatchNorm(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            SyncBatchNorm(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


@MODELS.register_module()
class IR_SENet(BaseModule):
    def __init__(
            self,
            pretrained: Optional[str] = None,
            num_layers: int = 50,
            mode: str = 'ir_se',
            out_indices: Sequence[int] = (3, )):

        super(IR_SENet, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'

        self.pretrained = pretrained
        self.out_indices = out_indices

        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      SyncBatchNorm(64),
                                      PReLU(64))
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        self.body = ModuleList(modules)

    def init_weights(self):

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            pretrained_state_dict = _load_checkpoint(self.pretrained, map_location='cpu')
            for name, value in self.state_dict().items():
                if name in pretrained_state_dict.keys():
                    value.data.copy_(pretrained_state_dict[name])

            logger.info(f"Loaded {len(self.state_dict())} weights from the pretrained model.")

        else:
            super().init_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.maxpool(x)
        outs = []
        for i, module in enumerate(self.body):
            x = module(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)