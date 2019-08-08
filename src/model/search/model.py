import torch
import torch.nn as nn
import torch.nn.functional as F
from model.operations import *
from model.genotypes import PRIMITIVES
from model.search.cell import Cell
import utils.utils as utils


class Network(nn.Module):
    """
    Searching Network
    """
    def __init__(self, C, n_colors, scale, n_cells, n_nodes=4, multiplier=4):
        """
        Args:
            C: number of initial channels
               (the input channels of the first Cell)
            n_colors: number of pic colors(3, RGB by default)
            scale: scale factor for the Upsampler
            n_cells: the whole number of Cells
            n_nodes: the whole number of intermediate nodes in one Cell
            multiplier: number of intermediate nodes that will be concat
                        (default is n_nodes)
        """
        super(Network, self).__init__()
        self._C = C
        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.multiplier = multiplier

        # what is stem_multiplier?
        self.stem_multiplier = 3

        C_curr = self.stem_multiplier * self._C

        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C

        self.cells = nn.ModuleList()
        for i in range(self.n_cells):
            # # In 1/3 and 2/3 location of Cells, set 2 Reduce Cells
            # # to reduce featuremap size and double channels.
            # if i in [n_cells//3, 2*n_cells//3]:
            #     C_curr *= 2
            #     reduction = True
            # else:
            #     reduction = False
            cell = Cell(self.n_nodes, self.multiplier,
                        C_prev_prev, C_prev, C_curr)
            self.cells.append(cell)
            C_curr_out = self.multiplier * C_curr
            C_prev_prev, C_prev = C_prev, C_curr_out

        self.upsampler = Upsampler(C_prev, C_prev, 3,
                                   stride=1, padding=1, scale=scale)
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(C_prev, n_colors, 3, padding=1, bias=False),
        )

    def forward(self, input, weights_normal):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, weights_normal)

        out = self.upsampler(s1)
        logits = self.channel_reducer(out)
        return logits
