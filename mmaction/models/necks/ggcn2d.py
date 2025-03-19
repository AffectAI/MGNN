from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid, Conv2d, AdaptiveAvgPool2d

from mmcv.cnn import ConvModule

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor


class ResGatedGraphConv2d(MessagePassing):
    r"""The residual gated graph convolutional operator from the
    `"Residual Gated Graph ConvNets" <https://arxiv.org/abs/1711.07553>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \eta_{i,j} \odot \mathbf{W}_2 \mathbf{x}_j

    where the gate :math:`\eta_{i,j}` is defined as

    .. math::
        \eta_{i,j} = \sigma(\mathbf{W}_3 \mathbf{x}_i + \mathbf{W}_4
        \mathbf{x}_j)

    with :math:`\sigma` denoting the sigmoid function.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        act (callable, optional): Gating function :math:`\sigma`.
            (default: :meth:`torch.nn.Sigmoid()`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        conv_kernel_size=3,
        conv_cfg=None,
        norm_cfg=dict(type='GN', num_groups=8),
        act: Optional[Callable] = Sigmoid(),
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        avg_pooled: bool = True,
        single_value_edge: bool = False,
        **kwargs,
    ):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.edge_dim = edge_dim
        self.root_weight = root_weight
        self.avg_pooled = avg_pooled
        self.conv_kernel_size = conv_kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.single_value_edge = single_value_edge
        padding = (self.conv_kernel_size - 1) // 2

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        assert (self.single_value_edge and self.avg_pooled) is False
        edge_dim = edge_dim if edge_dim is not None else 0

        self.lin_key = Conv2d(in_channels[1] + edge_dim, out_channels, kernel_size=1, padding=0)
        self.lin_query = Conv2d(in_channels[0] + edge_dim, out_channels, kernel_size=1, padding=0)
        self.lin_value = Conv2d(in_channels[0] + edge_dim, out_channels, kernel_size=1, padding=0)

        # self.lin_cat = Conv2d(out_channels * 2, out_channels, kernel_size=1, padding=0)

        if root_weight:
            self.lin_skip = Conv2d(in_channels[1], out_channels, kernel_size=1, padding=0, bias=False)
        else:
            self.register_parameter('lin_skip', None)

        if bias:
            self.bias = Parameter(torch.empty(out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        if self.avg_pooled:
            self.pool = AdaptiveAvgPool2d(1)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()

        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:


        if isinstance(x, Tensor):
            # x = self.conv(x)
            x: PairTensor = (x, x)

        # In case edge features are not given, we can compute key, query and
        # value tensors in node-level space, which is a bit more efficient:
        if self.edge_dim is None:
            k = self.lin_key(x[1])
            q = self.lin_query(x[0])
            v = self.lin_value(x[0])
        else:
            k, q, v = x[1], x[0], x[0]

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None)

        if self.root_weight:
            out = out + self.lin_skip(x[1])
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, k_i: Tensor, q_j: Tensor, v_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        assert (edge_attr is not None) == (self.edge_dim is not None)
        if edge_attr is not None:
            if not self.avg_pooled:
                assert edge_attr.dim() == 4, \
                    'if not conduct average pooling, edge_attr should has shape as [B, N, H, W].'

            k_i = self.lin_key(torch.cat([k_i, edge_attr], dim=1))
            q_j = self.lin_query(torch.cat([q_j, edge_attr], dim=1))
            v_j = self.lin_value(torch.cat([v_j, edge_attr], dim=1))

        if self.avg_pooled:
            weight = self.act(self.pool(k_i + q_j))
        elif self.single_value_edge:
            weight = self.act(k_i + q_j).mean((1, 2, 3), keepdim=True)
        else:
            weight = self.act(k_i + q_j)
        return weight * v_j

        # if self.avg_pooled:
        #     weight = self.act(self.pool(self.lin_cat(torch.cat([k_i, q_j], dim=1))))
        # else:
        #     weight = self.act(self.lin_cat(torch.cat([k_i, q_j], dim=1)))
        # return weight * v_j


        # h, w = k_i.shape[-2], k_i.shape[-1]
        # k_i_ = k_i.view(k_i.shape[0], k_i.shape[1], -1, 1)
        # q_j_ = q_j.view(q_j.shape[0], q_j.shape[1], 1, -1)
        # v_j_ = v_j.view(v_j.shape[0], v_j.shape[1], -1, 1)
        # return torch.matmul(self.act(torch.matmul(k_i_, q_j_)), v_j_).view(v_j.shape[0], v_j.shape[1], h, w)


if __name__ == '__main__':
    x = torch.randn([4, 256, 32, 32])
    # x = torch.tensor([[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.], [4., 4., 4., 4.]])
    edge_index = torch.tensor([[1, 2, 3, 3, 1], [0, 0, 0, 1, 3]])
    edge_attr = torch.tensor([[1., 1., 1., 1., 1.], [2., 2., 2., 2., 3.], [3., 3., 3., 3., 3.],
                              [4., 4., 4., 4., 5.], [4., 4., 4., 4., 5.]])

    model = ResGatedGraphConv2d(256, 2, edge_dim=None, avg_pooled=True)
    out = model(x, edge_index)
    print(out.shape)
