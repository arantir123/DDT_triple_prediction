"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args, n_nodes, feat_dim, is_add = 0):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    # default: act=None
    if not args.act:
        # define a constant mapping lambda x: x
        act = lambda x: x
    else:
        # obtain required activation function
        act = getattr(F, args.act)

    # for cases in which HyboNet is not used for GCN_add (assert args.num_layers > 1 in HyboNet function)
    if not is_add:
        # args.num_layers = 2
        acts = [act] * (args.num_layers - 1)
        dims = [feat_dim] + ([args.dim] * (args.num_layers - 1))
        if args.task in ['lp', 'rec']:
            dims += [args.dim]
            acts += [act]
            # specify the curvature for each Lorentz GCN layer
            n_curvatures = args.num_layers
        else:
            n_curvatures = args.num_layers - 1
    # for cases in which HyboNet is used for GCN_add (one LorentzGraphConvolution layer in total)
    else:
        acts = [act] * (is_add)
        # keep hidden state dim unchanged
        dims = [args.dim] + ([args.dim] * is_add)
        n_curvatures = is_add

    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]

    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias, scale=10)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

# Eeach Lorentz GCN layer contains one fully Lorentz linear layer and one Lorentz centroid layer
class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """
    # Lorentz fucntion, in_dim, out_dim, 1, 0, 0, 0, nonlin=act if i != 0 else None
    # default act: constant function
    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, local_agg, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        # manifold=Lorentz function, out_dim, dropout=0, use_bias=0, local_agg=0
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att, local_agg)
        # self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        # h = self.hyp_act.forward(h)
        output = h, adj
        return output

# LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()

        self.manifold = manifold # Lorentzian operation class, calling functions in lmath
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        # print(in_features,out_features)
        self.bias = bias
        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        # scale=10
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale) # trainable scale value

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)

        # the restriction implementation of fully Lorentz linear layer
        x = self.weight(self.dropout(x))

        # torch.narrow(input, dim, start, length) → Tensor
        # spatial dimension calculation
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        # time dimension calculation
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1

        scale = (time * time - 1) / (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)

        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    # restricted initialization of fully Lorentz linear layer
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features

        nn.init.uniform_(self.weight.weight, -stdv, stdv)

        with torch.no_grad():
            for idx in range(0, self.in_features, step):

                self.weight.weight[:, idx] = 0

        if self.bias:
            # Fills the input Tensor with the value \text{val}val.
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(Module):
    """
    Lorentz aggregation layer.
    """
    # manifold=Lorentz function, out_dim, dropout=0, use_bias=0, local_agg=0
    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        # default: 0
        self.dropout = dropout
        # default: 0
        self.local_agg = local_agg
        # default: 0
        self.use_att = use_att
        # use_att: QKV based attention mechanism
        if self.use_att:
            # self.att = DenseAtt(in_features, dropout)
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        # x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            # transductive based averaging feature aggregation
            support_t = torch.spmm(adj, x)
        # output = self.manifold.expmap0(support_t, c=self.c)

        denom = (-self.manifold.inner(None, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        output = support_t / denom
        # output: Lorentz centroid calculation results
        return output

    def attention(self, x, adj):
        pass


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
