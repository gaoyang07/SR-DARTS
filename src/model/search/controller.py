import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.operations import *
from model.genotypes import PRIMITIVES as op_names
from model.genotypes import Genotype
from torch.nn.parallel._functions import Broadcast
from model.search.model import Network as Network


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class NetworkController(nn.Module):
    """Network Controller for searching process"""

    def __init__(self, args, loss):
        super().__init__()
        self.args = args
        self._criterion = loss
        self.n_cells = args.layers
        self._steps = 4
        self._multiplier = 4

        if args.use_temp:
            self.temperature = args.initial_temp
        else: self.temperature = 1.0
        self.temperature = torch.tensor(self.temperature).cuda()

        self.use_sparsemax = args.use_sparsemax
        self.use_concrete = args.use_concrete
        if self.use_sparsemax:
            from utils.sparsemax import Sparsemax
            self.score_func = Sparsemax()
        else:
            self.score_func = F.softmax

        if self.use_concrete:
            raise NotImplementedError

        self._initialize_alphas()

        self.device_ids = args.gpus
        if args.gpus is None:
            device_ids = list(range(torch.cuda.device_count()))
            self.device_ids = device_ids

        self.network = Network(self.args, self._criterion)

    def forward(self, x):
        if not self.use_concrete:
            weights_normal = self.score_func(
                self.alphas_normal / self.temperature, dim=-1)
        else:
            raise NotImplementedError

        if len(self.device_ids) == 1:
            return self.network(x, weights_normal)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.network, self.device_ids)
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
        num_ops = len(op_names)

        # self.alphas_normal = nn.ParameterList()
        # for i in range(self.l)
        self.alphas_normal = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
        ]

    def scores(self):
        scores = []
        for alpha in self.alphas:
            scores.append(self.score_func(alpha).data)
        return scores

    def _gumbel_softmax_sample(self, logits):
        def sample_gumbel(shape, eps=1e-20):
            U = torch.rand(shape).cuda()
            return -torch.log(-torch.log(U + eps) + eps)

        y = logits + sample_gumbel(logits.size())
        return F.softmax(y / self.temperature, dim=-1)

    def sample(self):
        samples = []
        for alpha in self.alphas:
            if self.use_concrete:
                sample = self._gumbel_softmax_sample(alpha)
            else:
                sample = self.score_func(alpha, dim=-1)
            samples.append(sample)
        return samples

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if k != op_names.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != op_names.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((op_names[k_best], j))
                start = end
                n += 1
            return gene

        # scores = self.scores()
        # genes = []
        # for score in scores:
        #     genes.append(_parse(score.cpu().numpy()))

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype
