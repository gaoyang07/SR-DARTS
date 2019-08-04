from __future__ import division

import numpy as np
import torch
from torch import nn
from .base import _BaseBatchProjection


def project_simplex(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).cuda()
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)
    return w


def sparsemax_grad(dout, w_star):
    supp = w_star > 0
    masked = dout.masked_select(supp)
    masked -= masked.sum() / supp.sum().float()
    out = dout.new(dout.size()).zero_()
    out[supp] = masked
    return(out)


class SparsemaxFunction(_BaseBatchProjection):

    def project(self, x):
        return project_simplex(x)

    def project_jv(self, dout, y_star):
        return sparsemax_grad(dout, y_star)


class Sparsemax(nn.Module):

    def forward(self, x, temperature, dim=-1):
        sparsemax = SparsemaxFunction()
        return sparsemax(x)
