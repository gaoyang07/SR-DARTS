import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.operations import *
from model.genotypes import PRIMITIVES
from model.genotypes import Genotype
from torch.nn.parallel._functions import Broadcast
from model.search.model import Network as Network


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class NetworkController(nn.Module):
    """Support multi-gpu"""

    def __init__(self, args, loss):
        super().__init__()
        self.args = args
        # self.loss = loss
        self._criterion = loss
        self._steps = 4
        self._multiplier = 4
        self.temperature = args.initial_temp

        self.device_ids = args.gpus
        print("self.device_ids: {}".format(self.device_ids))
        if args.gpus is None:
            device_ids = list(range(torch.cuda.device_count()))
            self.device_ids = device_ids

        self._initialize_alphas()

        self.network = Network(self.args, self._criterion)

    def forward(self, x):
        weights_normal = F.softmax(
            self.alphas_normal / self.temperature, dim=-1)

        if len(self.device_ids) == 1:
            return self.network(x, weights_normal)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.network, self.device_ids)
        print("replicas.len: {}".format( len(replicas) ))
        print("list(zip(xs, wnormal_copies)): {}".format( len(list(zip(xs, wnormal_copies))) ))
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def temp_update(self, epoch):
        self.temperature = self.temperature * \
            math.pow(self.args.temp_beta, epoch)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.alphas_reduce = Variable(
        #     1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            # self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def new(self):
        model_new = Network(self.args, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        # gene_reduce = _parse(
        #     F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            # reduce=gene_reduce, reduce_concat=concat
        )
        return genotype