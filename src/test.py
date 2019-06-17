import os
import sys
import glob
import time
import numpy as np
import torch
import utils
import tqdm
import logging
import argparse
import torch.nn as nn
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
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, args.layers, args.scale, genotype)
    utils.load(model, args.model_path)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.L1Loss()
    criterion = criterion.cuda()

    test_loader = DataLoader(args).test_loader

    model.drop_path_prob = args.drop_path_prob

    # test_acc, test_loss = infer(test_loader, model, criterion)
    # logging.info('test_acc %f', test_acc)

    infer(test_loader, model, criterion)


def infer(test_queue, model, criterion):
    # objs = utils.AverageMeter()
    eval_index = utils.AverageMeter()
    model.eval()

    for idx_data, d in enumerate(test_queue):
        eval_index = utils.AverageMeter()
        for step, (_input, _target, _, idx_scale) in enumerate(d):
            _input = _input.clone().detach().requires_grad_(False).cuda()
            _target = _target.clone().detach().requires_grad_(False).cuda(non_blocking=True)

            logits = model(_input)
            logits = utils.quantize(logits, args.rgb_range)
            psnr = utils.calc_psnr(logits, _target, idx_scale,
                                   args.rgb_range, is_search=False)
            n = _input.size(0)
            # objs.update(loss.item(), n)
            eval_index.update(psnr, n)
            # print("In data {}: {}".format(idx_data, psnr))
            # loss = criterion(logits, _target)
        logging.info('{}: PSNR: {}'.format(idx_data, eval_index.avg))
    # return eval_index.avg


if __name__ == '__main__':
    main()
