import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint

using_ckpt = False


class IBasicBlock(nn.Module):

    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True)):
        super(IBasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = build_norm_layer(norm_cfg, inplanes)[1]
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.prelu = nn.PReLU(planes)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes)[1]

        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


@MODELS.register_module()
class IResNet(BaseModule):
    fc_scale = 7 * 7

    arch_settings = {
        18: (IBasicBlock, [2, 2, 2, 2]),
        34: (IBasicBlock, [3, 4, 6, 3]),
        50: (IBasicBlock, [3, 4, 14, 3]),
        100: (IBasicBlock, [3, 13, 30, 3])
    }

    def __init__(
            self,
            depth: int,
            pretrained: Optional[str] = None,
            num_stages: int = 4,
            out_indices: Sequence[int] = (3,),
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            fp16=False,
            conv_cfg: ConfigType = dict(type='Conv'),
            norm_cfg: ConfigType = dict(type='BN2d', requires_grad=True),
            init_cfg: Optional[Union[Dict, List[Dict]]] = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', layer='BatchNorm2d', val=1.)
            ]
    ):
        super(IResNet, self).__init__(init_cfg=init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for iresnet')
        self.depth = depth
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages

        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = build_conv_layer(
            conv_cfg,
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = build_norm_layer(norm_cfg, self.inplanes)[1]
        self.prelu = nn.PReLU(self.inplanes)

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self.layer1 = self._make_layer(self.block, 64, self.stage_blocks[0], stride=2)
        self.layer2 = self._make_layer(self.block,
                                       128,
                                       self.stage_blocks[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.block,
                                       256,
                                       self.stage_blocks[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.block,
                                       512,
                                       self.stage_blocks[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def init_weights(self):
        if self.pretrained:
            self.init_cfg = dict(
                type='Pretrained', checkpoint=self.pretrained)
        super().init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)

            outs = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.out_indices:
                    outs.append(x)

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)