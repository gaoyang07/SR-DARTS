import torch
import torch.nn as nn
from model.operations import *
from model import genotypes as genotypes
from model.genotypes import PRIMITIVES
from model.genotypes import Genotype
from model.train.cell import Cell


# def make_model(args, parent=False):
#     return Network(args)

class Network(nn.Module):
    """Training Network"""
    def __init__(self, args):
        super(Network, self).__init__()

        self.args = args
        self.C = args.init_channels
        self._layers = args.layers
        self._scale = args.scale[0]
        self.drop_path_prob = args.drop_path_prob
        self.genotype = eval("genotypes.%s" % args.arch)
        # self._auxiliary = auxiliary
        self.stem_multiplier = 3

        C_curr = self.stem_multiplier * self.C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(self._layers):
            # if i in [self._layers // 3, 2 * self._layers//3]:
            #     C_curr *= 2
            #     reduction = True
            # else:
            #     reduction = False
            reduction = False
            cell = Cell(self.genotype, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            # if i == 2*layers//3:
            #     C_to_auxiliary = C_prev

        # if auxiliary:
        #     self.auxiliary_head = AuxiliaryHeadSR(
        #         C_to_auxiliary, num_classes)

        self.upsampler = Upsampler(C_prev, C_prev, 3,
                                   stride=1, padding=1, scale=self._scale)
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(C_prev, args.n_colors, 3, padding=1, bias=False)
        )

    def forward(self, input):
        # logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            # if i == 2*self._layers//3:
            #     if self._auxiliary and self.training:
            #         logits_aux = self.auxiliary_head(s1)
        out = self.upsampler(s1)
        logits = self.channel_reducer(out)
        return logits
