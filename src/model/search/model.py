import torch
import torch.nn as nn
import torch.nn.functional as F
from model.operations import *
from model.genotypes import PRIMITIVES
from model.search.cell import Cell
import utils.utils as utils


class Network(nn.Module):
    """Searching Network"""
    def __init__(self, args, loss):
        super(Network, self).__init__()
        self.args = args
        self._C = args.init_channels
        self._layers = args.layers
        self._scale = args.scale[0]
        self._criterion = loss
        self._steps = 4
        self._multiplier = 4
        self.stem_multiplier = 3

        C_curr = self.stem_multiplier * self._C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
        self.cells = nn.ModuleList()
        for i in range(self._layers):
            self.cells.append(
                Cell(self._steps, self._multiplier, C_prev_prev, C_prev, C_curr)
            )
            C_prev_prev, C_prev = C_prev, self._multiplier * C_curr

        self.upsampler = Upsampler(C_prev, C_prev, 3,
                                   stride=1, padding=1, scale=self._scale)
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(C_prev, args.n_colors, 3, padding=1, bias=False),
        )

    def forward(self, input, weights_normal):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, weights_normal)

        out = self.upsampler(s1)
        logits = self.channel_reducer(out)
        return logits
