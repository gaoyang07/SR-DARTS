import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn

from model.model import Network
from model.common import *
from model import genotypes
from configs.test_configs import args
from data.dataloader import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

args.save = '{}test-{}-{}'.format(args.save,
                                      args.note, time.strftime("%Y%m%d-%H%M%S"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES,
                    args.layers, args.auxiliary, genotype)
    model = model.cuda()
    utils.load(model, args.model_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_transforms_cifar10(args)
    test_data = dset.CIFAR10(root=args.data, train=False,
                             download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %f %f', step,
                         objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
