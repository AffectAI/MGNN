from warnings import warn
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn import build_norm_layer, ConvModule
from torch_geometric.nn import GCNConv, GATConv, ResGatedGraphConv
from .ggcn2d import ResGatedGraphConv2d
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from mmengine.model.weight_init import constant_init, normal_init, xavier_init


class GraphNeck(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 graph_layer=None,
                 num_layers=1,
                 is_graph2d=False,
                 in_channels=256,
                 out_channels=256,
                 norm_cfg=dict(type='GN', num_groups=8), # dict(type='GN', num_groups=8)
                 temporal_choose=None,
                 init_cfg=None):

        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(GraphNeck, self).__init__(init_cfg)

        graph_conv = nn.ModuleList()
        graph_norm = nn.ModuleList()
        if graph_layer is None and not is_graph2d:
            for i in range(num_layers):
                graph_conv.append(GCNConv(out_channels, out_channels))
                graph_norm.append(build_norm_layer(norm_cfg, out_channels)[1])
        if graph_layer is not None and not is_graph2d:
            for i in range(num_layers):
                graph_conv.append(self._get_graph_layer(graph_layer, out_channels))
                graph_norm.append(build_norm_layer(norm_cfg, out_channels)[1])
        if graph_layer is not None and is_graph2d:
            for i in range(num_layers):
                graph_conv.append(self._get_graph_layer(graph_layer, out_channels))
                graph_norm.append(build_norm_layer(norm_cfg, out_channels)[1])

        self.conv_1x1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg)


        if not is_graph2d:
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.graph_conv = graph_conv
        self.graph_norm = graph_norm
        self.act = nn.ReLU(True)
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.is_graph2d = is_graph2d
        self.temporal_choose = temporal_choose

    @staticmethod
    def _get_graph_layer(graph_layer, in_channels):
        if graph_layer in ['GCNConv', 'ResGatedGraphConv', 'GCNConv2d']:
            return globals()[graph_layer](in_channels, in_channels)
        elif graph_layer in ['GATConv', 'GATConv2d']:
            return globals()[graph_layer](
                in_channels, in_channels,
                heads=8, concat=False, alpha_sum=False)
        elif graph_layer == 'ResGatedGraphConv2d':
            return globals()[graph_layer](
                in_channels, in_channels, avg_pooled=False, single_value_edge=False)
        else:
            raise NotImplementedError('Please check the name of the graph conv method.')

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x, batch_idx):
        # x shape: [N, C, H, W] -> [B, T, C, H, W]
        x = self.conv_1x1(x)

        feats = []
        for i in range(len(batch_idx)):
            feat = x[:batch_idx[i], ...]
            x = x[batch_idx[i]:, ...]
            feats.append(feat)

        # x = x.reshape((-1, num_segs) + x.shape[1:])

        batch_size = len(feats)
        batch_feats = []
        for i in range(batch_size):
            feat = feats[i]
            graph_edge_index = self._get_edge_index(feat.size(0)).to(feat.device)
            graph_feats = Data(x=feat, edge_index=graph_edge_index)
            batch_feats.append(graph_feats)

        graph_data = Batch.from_data_list(batch_feats)
        graph_feats = graph_data.x

        if not self.is_graph2d:
            graph_feats = self.pool(graph_feats).flatten(1)
        for i in range(self.num_layers):
            graph_feats = self.graph_conv[i](graph_feats, graph_data.edge_index)
            graph_feats = self.graph_norm[i](graph_feats)
            graph_feats = self.act(graph_feats)
        if not self.is_graph2d:
            graph_feats = graph_feats.unsqueeze(-1).unsqueeze(-1)

        temp_out = []
        for i in range(len(batch_idx)):
            feat = graph_feats[:batch_idx[i], ...]
            graph_feats = graph_feats[batch_idx[i]:, ...]

            if self.temporal_choose is None or self.temporal_choose == 'mean':
                feat = feat.mean(0)
            elif self.temporal_choose == 'last_frame':
                feat = feat[-1, ...]
            else:
                raise NotImplementedError('temporal_choose should be mean or last_frame.')

            temp_out.append(feat)

        temp_out = torch.stack(temp_out, dim=0)

        return temp_out

    def _get_edge_index(self, num_classes):
        node_id = [i for i in range(num_classes)]

        u = []
        v = []
        for i, n in enumerate(node_id):
            u += [n] * len(node_id[i:])
            v += node_id[i:]

        return torch.tensor([u, v])








