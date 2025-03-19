from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter, ModuleList
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


##################################  MobileFaceNet #############################################################

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


@MODELS.register_module()
class MobileFaceNet(BaseModule):
    def __init__(
            self,
            pretrained: Optional[str] = None):
        super(MobileFaceNet, self).__init__()
        self.pretrained = pretrained

        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        # self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))

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
        outs = []
        out = self.conv1(x)
        out = self.conv2_dw(out)
        outs.append(out)

        out = self.conv_23(out)
        out = self.conv_3(out)
        outs.append(out)

        out = self.conv_34(out)
        out = self.conv_4(out)
        outs.append(out)

        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        # out = self.conv_6_dw(out)
        outs.append(out)

        return tuple(outs)