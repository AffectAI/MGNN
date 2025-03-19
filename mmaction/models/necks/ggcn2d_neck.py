from typing import Optional, Tuple, Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import constant_init, normal_init, xavier_init

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType, SampleList
from .gnn_neck import GraphNeck


class DownSample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]] = (1, 1),
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = 0,
            groups: int = 1,
            bias: Union[bool, str] = False,
            conv_cfg: ConfigType = dict(type='Conv2d'),
            norm_cfg: OptConfigType = None,
            act_cfg: OptConfigType = None,
            downsample_position: str = 'after',
            downsample_kernel: Union[int, Tuple[int]] = (3, 3)
    ) -> None:
        super().__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        assert downsample_position in ['before', 'after']
        self.downsample_position = downsample_position

        self.pool = ConvModule(
            out_channels,
            out_channels,
            downsample_kernel,
            stride=2,
            padding=downsample_kernel // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        if self.downsample_position == 'before':
            x = self.pool(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.pool(x)
        return x


class AuxHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            loss_weight: float = 0.5,
            temporal_choose: Optional[str] = None,
            loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d', requires_grad=True)
    ) -> None:
        super().__init__()

        self.conv = ConvModule(
            in_channels,
            in_channels * 2, (1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_channels * 2, out_channels)
        self.loss_cls = MODELS.build(loss_cls)
        self.temporal_choose = temporal_choose

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def loss(self,
             x: torch.Tensor,
             num_segs: int,
             data_samples: Optional[SampleList], ) -> dict:
        """Calculate auxiliary loss."""
        x = self(x, num_segs)
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(x.device)
        labels = labels.squeeze()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        loss = self.loss_weight * self.loss_cls(x, labels)
        return loss

    def forward(self, x: torch.Tensor, num_segs: int) -> torch.Tensor:
        """Auxiliary head forward function."""
        x = self.conv(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = x.reshape((-1, num_segs) + x.shape[1:])
        if self.temporal_choose is None or self.temporal_choose == 'mean':
            x = x.mean(1)
        elif self.temporal_choose == 'last_frame':
            x = x[:, -1, ...]
        else:
            raise NotImplementedError('temporal_choose should be mean or last_frame.')
        x = self.dropout(x)
        x = self.fc(x)
        return x


@MODELS.register_module()
class MGNNHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            is_aux: bool = True,
            dropout_ratio: float = 0.4,
            loss_weight: float = 1.0,
            loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d', requires_grad=True)
    ) -> None:
        super().__init__()

        self.conv = ConvModule(
            in_channels,
            in_channels*2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.is_aux = is_aux
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels*2, num_classes)
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(dropout_ratio)

        self.loss_cls = MODELS.build(loss_cls)

        if not is_aux:
            self.weight = nn.Parameter(torch.FloatTensor(torch.ones(4, requires_grad=True)))

        self.init_weights()

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def loss(self,
             x: torch.Tensor,
             data_samples: Optional[SampleList],
             **kwargs) -> dict:
        """Calculate auxiliary loss."""
        x = self(x, **kwargs)
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(x.device)
        # labels = labels.squeeze()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        loss = self.loss_weight * self.loss_cls(x, labels)

        if self.is_aux:
            return loss
        else:
            return {'loss_cls': loss}

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """

        for data_sample, score in zip(data_samples, cls_scores):
            data_sample.set_pred_score(score)
        return data_samples

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Auxiliary head forward function."""
        if not self.is_aux:
            outs = []
            for i in range(len(x)):
                out = self.conv(x[i])
                out = self.avg_pool(out).squeeze(-1).squeeze(-1)
                out = self.dropout(out)
                outs.append(out)
            outs = torch.stack(outs, dim=0)
            weight = F.normalize(self.weight, p=2, dim=-1)
            return self.fc(torch.einsum('sbc, s -> bc', outs, weight))

        x = self.conv(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class StageFusion(nn.Module):
    def __init__(self, in_channels, num_stages,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d', requires_grad=True)
                 ):
        super(StageFusion, self).__init__()
        self.num_stages = num_stages
        self.down_sample_layers = nn.ModuleList()
        self.spat_fusion_layers = nn.ModuleList()
        self.temp_fusion_layers = nn.ModuleList()
        for i in range(num_stages - 1):
            down_op = ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            spat_op = ConvModule(
                in_channels * 2,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            temp_op = GraphNeck(
                graph_layer='ResGatedGraphConv2d',
                num_layers=1,
                is_graph2d=True,
                in_channels=in_channels,
                out_channels=in_channels,
                norm_cfg=norm_cfg)
            self.down_sample_layers.append(down_op)
            self.spat_fusion_layers.append(spat_op)
            self.temp_fusion_layers.append(temp_op)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self,
                x: List[torch.Tensor],
                num_segs: int):
        temp_fuse = None
        temp_fuses = []
        for i in range(self.num_stages - 1):
            feat = x[i] if i == 0 else temp_fuse
            feat_down = self.down_sample_layers[i](feat)
            spat_fuse = self.spat_fusion_layers[i](
                torch.cat([feat_down, x[i + 1]], dim=1))
            temp_fuse = self.temp_fusion_layers[i](spat_fuse, num_segs)
            temp_fuses.append(temp_fuse)
        return temp_fuses


class BottleNeck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 ex_ratio: float = 2.0,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d', requires_grad=True)):
        super(BottleNeck, self).__init__()
        self.conv1 = ConvModule(
            in_channels,
            int(in_channels * ex_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            int(in_channels * ex_ratio),
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        out = self.conv2(self.conv1(x)) + x
        return out


class MultiScaleFFM(nn.Module):
    def __init__(self,
                 in_channels: Tuple[int],
                 out_channels: int,
                 num_stages: int,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d', requires_grad=True)):
        super(MultiScaleFFM, self).__init__()
        self.num_stages = num_stages
        self.stage_1 = nn.ModuleList(
            [ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg) for i in range(num_stages)])

        self.stage_2 = nn.ModuleList(
            [ConvModule(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg) for _ in range(num_stages - 1)])

        self.stage_f1 = nn.ModuleList(
            [BottleNeck(out_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg) for _ in range(num_stages - 1)])

        self.stage_3 = nn.ModuleList(
            [ConvModule(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg) for _ in range(num_stages - 2)])

        self.stage_f2 = nn.ModuleList(
            [BottleNeck(out_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg) for _ in range(num_stages - 1)])

        self.down_sample = nn.Upsample(
            scale_factor=0.5, mode='nearest')
        self.up_sample = nn.Upsample(
            scale_factor=2.0, mode='nearest')

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x: List[torch.Tensor]):
        x1 = []
        for i in range(self.num_stages):
            x_ = self.stage_1[i](x[i])
            x1.append(x_)

        down_fuses = []
        down_fuse = None
        for i in reversed(range(1, self.num_stages)):
            feat = x1[i] if i == self.num_stages - 1 else down_fuse
            down_fuse = self.up_sample(feat)
            down_fuse = self.stage_2[i - 1](torch.cat([x1[i - 1], down_fuse], dim=1))
            down_fuse = self.stage_f1[i - 1](down_fuse)
            down_fuses.append(down_fuse)

        down_fuses = down_fuses[::-1]
        up_fuses = [down_fuses[0]]
        up_fuse = None
        for i in range(self.num_stages - 2):
            feat = down_fuses[i] if i == 0 else up_fuse
            up_fuse = self.down_sample(feat)
            up_fuse = self.stage_3[i](torch.cat([down_fuses[i + 1], up_fuse], dim=1))
            up_fuse = self.stage_f2[i](up_fuse + x1[i + 1])
            up_fuses.append(up_fuse)

        up_fuse = self.down_sample(up_fuses[-1])
        up_fuse = self.stage_f2[-1](up_fuse + x1[-1])
        up_fuses.append(up_fuse)

        return up_fuses


class SpatialGraph(nn.Module):
    def __init__(self,
                 in_channels: Tuple[int],
                 out_channels: int,
                 num_stages: int,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d', requires_grad=True)):
        super(SpatialGraph, self).__init__()
        self.num_stages = num_stages
        self.stage_1 = nn.ModuleList(
            [ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg) for i in range(num_stages)])

        self.stage_2 = nn.ModuleList(
            [nn.Conv2d(
                out_channels*2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False) for _ in range(6)])

        self.stage_s1 = nn.ModuleList(
            [nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False) for _ in range(4)])

        self.act = nn.Sigmoid()

        self.stage_3 = nn.ModuleList(
            [nn.Conv2d(
                out_channels*2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False) for _ in range(6)])

        self.norms = nn.ModuleList(
            [nn.GroupNorm(32, out_channels) for _ in range(4)])
        self.relu = nn.ReLU()

        self.down2 = nn.Upsample(
            scale_factor=0.5, mode='nearest')
        self.down4 = nn.Upsample(
            scale_factor=0.25, mode='nearest')
        self.down8 = nn.Upsample(
            scale_factor=0.125, mode='nearest')
        self.up2 = nn.Upsample(
            scale_factor=2.0, mode='nearest')
        self.up4 = nn.Upsample(
            scale_factor=4.0, mode='nearest')
        self.up8 = nn.Upsample(
            scale_factor=8.0, mode='nearest')

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x: List[torch.Tensor]):
        x1 = []
        for i in range(self.num_stages):
            x_ = self.stage_1[i](x[i])
            x1.append(x_)

        # top-down
        e_33 = self.act(self.stage_s1[0](x1[3]))
        e_22 = self.act(self.stage_s1[1](x1[2]))
        e_11 = self.act(self.stage_s1[2](x1[1]))
        e_00 = self.act(self.stage_s1[3](x1[0]))

        e_32 = self.act(self.stage_2[0](torch.cat([self.up2(x1[3]), x1[2]], dim=1)))
        e_31 = self.act(self.stage_2[1](torch.cat([self.up4(x1[3]), x1[1]], dim=1)))
        e_30 = self.act(self.stage_2[2](torch.cat([self.up8(x1[3]), x1[0]], dim=1)))
        e_21 = self.act(self.stage_2[3](torch.cat([self.up2(x1[2]), x1[1]], dim=1)))
        e_20 = self.act(self.stage_2[4](torch.cat([self.up4(x1[2]), x1[0]], dim=1)))
        e_10 = self.act(self.stage_2[5](torch.cat([self.up2(x1[1]), x1[0]], dim=1)))

        e_01 = self.act(self.stage_3[0](torch.cat([self.down2(x1[0]), x1[1]], dim=1)))
        e_02 = self.act(self.stage_3[1](torch.cat([self.down4(x1[0]), x1[2]], dim=1)))
        e_03 = self.act(self.stage_3[2](torch.cat([self.down8(x1[0]), x1[3]], dim=1)))
        e_12 = self.act(self.stage_3[3](torch.cat([self.down2(x1[1]), x1[2]], dim=1)))
        e_13 = self.act(self.stage_3[4](torch.cat([self.down4(x1[1]), x1[3]], dim=1)))
        e_23 = self.act(self.stage_3[5](torch.cat([self.down2(x1[2]), x1[3]], dim=1)))
        #
        # update nodes
        n_3 = self.relu(self.norms[0](e_33*(x1[3]) + e_03*(self.down8(x1[0])) + e_13*(self.down4(x1[1])) + e_23*(self.down2(x1[2]))))
        n_2 = self.relu(self.norms[1](e_22*(x1[2]) + e_32*(self.up2(x1[3])) + e_02*(self.down4(x1[0])) + e_12*(self.down2(x1[1]))))
        n_1 = self.relu(self.norms[2](e_11*(x1[1]) + e_31*(self.up4(x1[3])) + e_21*(self.up2(x1[2])) + e_01*(self.down2(x1[0]))))
        n_0 = self.relu(self.norms[3](e_00*(x1[0]) + e_30*(self.up8(x1[3])) + e_20*(self.up4(x1[2])) + e_10*(self.up2(x1[1]))))

        nodes = [n_0, n_1, n_2, n_3]
        return nodes


@MODELS.register_module()
class MGNN(nn.Module):
    def __init__(self,
                 in_channels: Tuple[int],
                 out_channels: int,
                 aux_head_cfg: OptConfigType = None,
                 temporal_choose: Optional[str] = None,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d', requires_grad=True)
                 ):
        super(MGNN, self).__init__()
        assert isinstance(in_channels, tuple)
        assert isinstance(out_channels, int)
        assert aux_head_cfg is None or isinstance(aux_head_cfg, dict)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = len(in_channels)
        self.temporal_choose = temporal_choose

        self.gnn_ops = nn.ModuleList()
        for i in range(self.num_stages):
            gnn = GraphNeck(graph_layer='ResGatedGraphConv2d',
                            num_layers=1,
                            is_graph2d=True,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            norm_cfg=norm_cfg,
                            temporal_choose=temporal_choose)
            self.gnn_ops.append(gnn)

        # self.spatial = MultiScaleFFM(in_channels,
        #                              out_channels,
        #                              self.num_stages,
        #                              conv_cfg=conv_cfg,
        #                              norm_cfg=norm_cfg)
        # self.temporal = MultiScaleFFM((out_channels,) * 4,
        #                               out_channels,
        #                               self.num_stages,
        #                               conv_cfg=conv_cfg,
        #                               norm_cfg=norm_cfg)
        self.spatial = SpatialGraph(in_channels,
                                    out_channels,
                                    self.num_stages,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg)
        self.temporal = SpatialGraph((out_channels,) * 4,
                                     out_channels,
                                     self.num_stages,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg)


        if aux_head_cfg is not None:
            self.aux_head = MGNNHead(out_channels,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     **aux_head_cfg)
        else:
            self.aux_head = None

    def forward(self,
                x: Tuple[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                batch_idx: list = None) -> tuple:

        # 空间多尺度融合模块
        loss_aux = dict()
        x = self.spatial(x)

        # 时间聚合模块
        temp_outs = []
        for i in range(self.num_stages):
            temp_out = self.gnn_ops[i](x[i], batch_idx)

            if self.aux_head is not None and data_samples is not None:
                loss = self.aux_head.loss(temp_out, data_samples)
                loss_aux[f'd{i}.loss_aux'] = loss

            temp_outs.append(temp_out)

        # 时间-空间多尺度融合模块
        temp_outs = self.temporal(temp_outs)
        return temp_outs, loss_aux
