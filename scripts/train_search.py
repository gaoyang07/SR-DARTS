import os
import sys
import time
import math
import glob
import torch
import utils
import logging
import torch.utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from torch.autograd import Variable

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from configs import args
from model.common import *
from model.model_search import Network
from model.architect import Architect
from data.dataloader import DataLoader as DataLoader

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


def main():

    # save the scripts and related parameters.
    args.save = '{}-search-{}'.format(args.save,
                                      time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()  # MSE loss is for SR task.
    criterion = criterion.cuda()

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    model = Network(args.init_channels, args.layers, args.scale, criterion)

    if args.checkpoint:
        load(model, args.model_path)
        logging.info("loading checkpoint: '{}'".format(args.model_path))
    model = model.cuda()

    logging.info("args = %s", args)
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    data_loader = DataLoader(args)
    train_loader = data_loader.train_loader
    valid_loader = data_loader.valid_loader
    test_loader = data_loader.test_loader

    optimizer_model = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # optimizer_arch = torch.optim.Adam(
    #     model.arch_parameters(),
    #     args.arch_learning_rate,
    #     betas=(0.5, 0.999),
    #     weight_decay=args.arch_weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    writer = SummaryWriter('{}/{}'.format(args.save, 'visualization'))
    start = time.time()
    best_valid_acc = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # temperature = args.initial_temp * np.exp(-args.anneal_rate * epoch)
        # temperature = args.initial_temp * math.pow(args.temp_beta, epoch)

        # train
        top1_acc, train_obj = train(
            train_loader, valid_loader, model, architect, criterion, optimizer_model, lr)

        logging.info('top1_acc %f, took %f sec',
                     top1_acc, time.time() - epoch_start)
        writer.add_scalar('search/acc', top1_acc, epoch)
        writer.add_scalar('search/loss', train_obj, epoch)
        writer.add_scalar('search/lr', lr, epoch)

        # valid
        valid_acc, _ = infer(valid_loader, model, criterion)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_genotype = genotype
            utils.save(model, os.path.join(args.save, 'weights.pt'))

        logging.info('valid_acc %f(best_acc %f), took %f sec',
                     valid_acc, best_valid_acc, time.time() - epoch_start)
        writer.add_scalar('valid/acc', valid_acc, epoch)

    logging.info('All epochs finished, took %f sec in total.',
                 time.time() - start)
    logging.info('Best genotype: {}'.format(best_genotype))
    writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):

    objs = utils.AverageMeter()
    eval_index = utils.AverageMeter()

    # for step, (input, target) in enumerate(valid_queue):
    for step, (lr, hr, _, idx_scale) in enumerate(train_queue):

        model.train()
        n = lr.size(0)

        input = torch.Tensor(input, requires_grad=False).cuda()
        target = torch.Tensor(
            target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = torch.Tensor(input_search, requires_grad=False).cuda()
        target_search = torch.Tensor(
            target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search,
                       lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()

        # output is the high-resolution image
        output = model(lr, idx_scale)
        loss = criterion(output, hr)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        psnr = utils.calc_PSNR(input, output)
        objs.update(loss.item(), n)
        eval_index.update(psnr, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step,
                         objs.avg, eval_index.avg)

    return eval_index.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    eval_index = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = torch.Tensor(input, requires_grad=False).cuda()
        target = torch.Tensor(
            target, requires_grad=False).cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        psnr = utils.calc_PSNR(input, output)
        n = input.size(0)
        objs.update(loss.item(), n)
        eval_index.update(psnr, n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step,
                         objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
