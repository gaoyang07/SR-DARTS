import torch
import torch.nn as nn
from torch.autograd import Variable
from model.common import drop_path
from model.operations import *
from model import genotypes as genotypes
from model.genotypes import PRIMITIVES
from model.genotypes import Genotype


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        # if reduction_prev:
        #     self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        # else:
        #     self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess0 = ReLUConv(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConv(C_prev, C, 1, 1, 0, affine=False)

        # if reduction:
        #     op_names, indices = zip(*genotype.reduce)
        #     concat = genotype.reduce_concat
        # else:
        #     op_names, indices = zip(*genotype.normal)
        #     concat = genotype.normal_concat
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat

        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            # stride = 2 if reduction and index < 2 else 1
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadSR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadSR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):

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
