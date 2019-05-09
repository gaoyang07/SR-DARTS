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
from torch.autograd import Variable

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
    best_valid_index = 0
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
        train_index, train_obj = train(
            train_loader, valid_loader, model, architect, criterion, optimizer_model, lr)

        logging.info('train_index %f, took %f sec',
                     train_index, time.time() - epoch_start)
        writer.add_scalar('search/index', train_index, epoch)
        writer.add_scalar('search/loss', train_obj, epoch)
        writer.add_scalar('search/lr', lr, epoch)

        # valid
        valid_index, _ = infer(valid_loader, model, criterion)
        if valid_index > best_valid_index:
            best_valid_index = valid_index
            best_genotype = genotype
            utils.save(model, os.path.join(args.save, 'weights.pt'))

        logging.info('valid_index %f(best_index %f), took %f sec',
                     valid_index, best_valid_index, time.time() - epoch_start)
        writer.add_scalar('valid/index', valid_index, epoch)

    logging.info('All epochs finished, took %f sec in total.',
                 time.time() - start)
    logging.info('Best genotype: {}'.format(best_genotype))
    writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):

    objs = utils.AverageMeter()
    eval_index = utils.AverageMeter()

    for step, (_input, _target, _, idx_scale) in enumerate(train_queue):

        model.train()
        n = _input.size(0)

        _input = Variable(_input, requires_grad=False).cuda()
        _target = Variable(_target, requires_grad=False).cuda(
            non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search, _, _ = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(
            target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(_input, _target, input_search, target_search,
                       lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()

        # output is the high-resolution image
        logits = model(_input)
        loss = criterion(logits, _target)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        psnr = utils.calc_psnr(logits, _target, int(args.scale),
                               args.rgb_range, dataset=args.data_train)
        objs.update(loss.item(), n)
        eval_index.update(psnr, n)

        if step % args.report_freq == 0:
            logging.info('train step: %03d   loss: %e    PSNR: %.2f', step,
                         objs.avg, eval_index.avg)

    return eval_index.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    eval_index = utils.AverageMeter()
    model.eval()

    for step, (_input, _target, _, idx_scale) in enumerate(valid_queue):

        _input = Variable(_input, requires_grad=False).cuda()
        _target = Variable(_target, requires_grad=False).cuda(
            non_blocking=True)

        logits = model(_input)
        loss = criterion(logits, _target)

        psnr = utils.calc_psnr(logits, _target, int(args.scale),
                               args.rgb_range, dataset=args.data_test)
        n = _input.size(0)

        objs.update(loss.item(), n)
        eval_index.update(psnr, n)

        if step % args.report_freq == 0:
            logging.info('valid step: %03d   loss: %e    PSNR: %.2f', step,
                         objs.avg, eval_index.avg)

    return eval_index.avg, objs.avg


if __name__ == '__main__':
    main()
