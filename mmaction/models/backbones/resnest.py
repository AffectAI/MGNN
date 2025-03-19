# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.logging import MMLogger
from mmaction.registry import MODELS
from mmengine.model import BaseModule, Sequential
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttentionConv2d(BaseModule):
    """Split-Attention Conv2d in ResNeSt.

    Args:
        in_channels (int): Number of channels in the input feature map.
        channels (int): Number of intermediate channels.
        kernel_size (int | tuple[int]): Size of the convolution kernel.
        stride (int | tuple[int]): Stride of the convolution.
        padding (int | tuple[int]): Zero-padding added to both sides of
        dilation (int | tuple[int]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to
            output channels.
        groups (int): Same as nn.Conv2d.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels. Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        dcn (dict): Config dict for DCN. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 radix=2,
                 reduction_factor=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 init_cfg=None):
        super(SplitAttentionConv2d, self).__init__(init_cfg)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.with_dcn = dcn is not None
        self.dcn = dcn
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if self.with_dcn and not fallback_on_stride:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            conv_cfg = dcn
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            channels * radix,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups * radix,
            bias=False)
        # To be consistent with original implementation, starting from 0
        self.norm0_name, norm0 = build_norm_layer(
            norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(
            None, channels, inter_channels, 1, groups=self.groups)
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(
            None, inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)

    @property
    def norm0(self):
        """nn.Module: the normalization layer named "norm0" """
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        batch = x.size(0)
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        gap = self.norm1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


class Bottleneck(_Bottleneck):
    """Bottleneck block for ResNeSt.

    Args:
        inplane (int): Input planes of this block.
        planes (int): Middle planes of this block.
        groups (int): Groups of conv2.
        base_width (int): Base of width in terms of base channels. Default: 4.
        base_channels (int): Base of channels for calculating width.
            Default: 64.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels in
            SplitAttentionConv2d. Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        kwargs (dict): Key word arguments for base class.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        """Bottleneck block for ResNeSt."""
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            print(base_width)
            print(base_channels)
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.avg_down_stride = avg_down_stride and self.conv2_stride > 1

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.with_modulated_dcn = False
        self.conv2 = SplitAttentionConv2d(
            width,
            width,
            kernel_size=3,
            stride=1 if self.avg_down_stride else self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            radix=radix,
            reduction_factor=reduction_factor,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            dcn=None)
        # delattr(self, self.norm2_name)

        if self.avg_down_stride:
            self.avd_layer = nn.AvgPool2d(3, self.conv2_stride, padding=1)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)

            if self.avg_down_stride:
                out = self.avd_layer(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class ResNeSt(ResNetV1d):
    """ResNeSt backbone.

    Args:
        groups (int): Number of groups of Bottleneck. Default: 1
        base_width (int): Base width of Bottleneck. Default: 4
        radix (int): Radix of SplitAttentionConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels in
            SplitAttentionConv2d. Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        kwargs (dict): Keyword arguments for ResNet.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 groups=1,
                 base_width=4,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        self.groups = groups
        self.base_width = base_width
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.avg_down_stride = avg_down_stride
        super(ResNeSt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            radix=self.radix,
            reduction_factor=self.reduction_factor,
            avg_down_stride=self.avg_down_stride,
            **kwargs)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            pretrained_state_dict = _load_checkpoint(self.pretrained, map_location='cpu')
            print(pretrained_state_dict.keys())
            print(self.state_dict().keys())
            raise
            pretrained_state_dict = pretrained_state_dict['state_dict']
            for key in list(pretrained_state_dict.keys()):
                if 'backbone' in key:
                    new_key = key.split('backbone.')[-1]
                    pretrained_state_dict[new_key] = pretrained_state_dict.pop(key)

            for name, value in self.state_dict().items():
                    value.data.copy_(pretrained_state_dict[name])

            logger.info(f"Loaded {len(self.state_dict())} weights from the pretrained model.")

        else:
            super().init_weights()
