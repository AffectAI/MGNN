from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer

from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import _load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import ConfigType


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride,
                 padding=0,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),):
        super().__init__()

        self.conv = ConvModule(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv(x)
        return x


class Block35(nn.Module):

    def __init__(self,
                 scale=1.0,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()

        self.scale = scale

        self.branch0 = ConvModule(256, 32, kernel_size=1, stride=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)

        self.branch1 = nn.Sequential(
            ConvModule(256, 32, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(32, 32, kernel_size=3, stride=1, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )

        self.branch2 = nn.Sequential(
            ConvModule(256, 32, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(32, 32, kernel_size=3, stride=1, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(32, 32, kernel_size=3, stride=1, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self,
                 scale=1.0,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()

        self.scale = scale

        self.branch0 = ConvModule(896, 128, kernel_size=1, stride=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)

        self.branch1 = nn.Sequential(
            ConvModule(896, 128, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(128, 128, kernel_size=(1, 7), stride=1, padding=(0,3),
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(128, 128, kernel_size=(7, 1), stride=1, padding=(3,0),
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self,
                 scale=1.0,
                 noReLU=False,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = ConvModule(1792, 192, kernel_size=1, stride=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)

        self.branch1 = nn.Sequential(
            ConvModule(1792, 192, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(192, 192, kernel_size=(1, 3), stride=1, padding=(0,1),
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(192, 192, kernel_size=(3, 1), stride=1, padding=(1,0),
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()

        self.branch0 = ConvModule(256, 384, kernel_size=3, stride=2, padding=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)

        self.branch1 = nn.Sequential(
            ConvModule(256, 192, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(192, 192, kernel_size=3, stride=1, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(192, 256, kernel_size=3, stride=2, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()

        self.branch0 = nn.Sequential(
            ConvModule(896, 256, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(256, 384, kernel_size=3, stride=2, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )

        self.branch1 = nn.Sequential(
            ConvModule(896, 256, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(256, 256, kernel_size=3, stride=2, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )

        self.branch2 = nn.Sequential(
            ConvModule(896, 256, kernel_size=1, stride=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(256, 256, kernel_size=3, stride=1, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            ConvModule(256, 256, kernel_size=3, stride=2, padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
        )

        self.branch3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


@MODELS.register_module()
class InceptionResnetV1(BaseModule):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(
            self,
            pretrained: Optional[str] = None,
            conv_cfg: ConfigType = dict(type='Conv'),
            norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
            act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained

        # Define layers
        self.conv2d_1a = ConvModule(3, 32, kernel_size=3, stride=2, padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.conv2d_2a = ConvModule(32, 32, kernel_size=3, stride=1, padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.conv2d_2b = ConvModule(32, 64, kernel_size=3, stride=1, padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2d_3b = ConvModule(64, 80, kernel_size=1, stride=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.conv2d_4a = ConvModule(80, 192, kernel_size=3, stride=1, padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.conv2d_4b = ConvModule(192, 256, kernel_size=3, stride=2, padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            pretrained_state_dict = _load_checkpoint(self.pretrained, map_location='cpu')
            # pretrained_state_dict = pretrained_state_dict['model']
            # print(self.state_dict().keys())
            # print(pretrained_state_dict.keys())
            # raise
            for name, value in self.state_dict().items():
                value.data.copy_(pretrained_state_dict[name])
            # 打印加载的权重信息
            logger.info(f"Loaded {len(self.state_dict())} weights from the pretrained model.")

        else:
            super().init_weights()

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        outs = []
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        outs.append(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        outs.append(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        outs.append(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        outs.append(x)
        return tuple(outs)