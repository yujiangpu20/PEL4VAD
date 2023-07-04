import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn.parameter import Parameter
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F
import numpy as np
import math


class DistanceAdj(nn.Module):
    def __init__(self, sigma, bias):
        super(DistanceAdj, self).__init__()
        # self.sigma = sigma
        # self.bias = bias
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)
        self.b.data.fill_(bias)

    def forward(self, batch_size, seq_len):
        arith = np.arange(seq_len).reshape(-1, 1)
        dist = pdist(arith, metric='cityblock').astype(np.float32)
        dist = torch.from_numpy(squareform(dist)).cuda()
        # dist = torch.exp(-self.sigma * dist ** 2)
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = torch.unsqueeze(dist, 0).repeat(batch_size, 1, 1)

        return dist


class TCA(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads, norm=None):
        super(TCA, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k  # same as dim_q
        self.n_heads = n_heads
        self.norm = norm

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)

        self.norm_fact = 1 / math.sqrt(dim_k)
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, mask, adj=None):
        Q = self.q(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        V = self.v(x).view(-1, x.shape[0], x.shape[1], self.dim_v // self.n_heads)

        if adj is not None:
            g_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact + adj
        else:
            g_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        l_map = g_map.clone()
        l_map = l_map.masked_fill_(mask.data.eq(0), -1e9)

        g_map = self.act(g_map)
        l_map = self.act(l_map)
        glb = torch.matmul(g_map, V).view(x.shape[0], x.shape[1], -1)
        lcl = torch.matmul(l_map, V).view(x.shape[0], x.shape[1], -1)

        alpha = torch.sigmoid(self.alpha)
        tmp = alpha * glb + (1 - alpha) * lcl
        if self.norm:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)  # l2 norm
        tmp = self.o(tmp).view(-1, x.shape[1], x.shape[2])
        return tmp
