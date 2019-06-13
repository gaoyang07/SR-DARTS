import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
import torch.utils
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from model.model import Network
from model.common import *
from model import genotypes
from configs.train_configs import args
from data.dataloader import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


def main():

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # save the scripts and related parameters.
    args.save = '{}eval-{}-{}'.format(args.save,
                                      args.note, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, args.layers, args.scale, genotype)
    if args.checkpoint:
        utils.load(model, args.model_path)
        logging.info("loading checkpoint: '{}'".format(args.model_path))
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.L1Loss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))

    data_loader = DataLoader(args)
    train_loader = data_loader.train_loader
    valid_loader = data_loader.valid_loader
    test_loader = data_loader.test_loader

    writer = SummaryWriter('{}/{}'.format(args.save, 'visual'))
    start = time.time()
    best_valid_acc = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        writer.add_scalar('train/lr', lr, epoch)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_loader, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('train/loss', train_obj, epoch)

        valid_acc, valid_loss = infer(valid_loader, model, criterion)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            utils.save(model, os.path.join(args.save, 'weights.pt'))

        logging.info('valid_acc %f(best_acc %f), took %f sec',
                     valid_acc, best_valid_acc, time.time() - epoch_start)
        writer.add_scalar('valid/acc', valid_acc, epoch)
        writer.add_scalar('valid/loss', valid_loss, epoch)

    logging.info('All epochs finished, took %f sec in total.',
                 time.time() - start)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    eval_index = utils.AverageMeter()
    model.train()

    for step, (_input, _target, _, idx_scale) in enumerate(train_queue):

        _input = _input.clone().detach().requires_grad_(False).cuda()
        _target = _target.clone().detach().requires_grad_(False).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(_input)
        loss = criterion(logits, _target)
        # if args.auxiliary:
        #     loss_aux = criterion(logits_aux, _target)
        #     loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        psnr = utils.calc_psnr(logits, _target, idx_scale,
                               args.rgb_range, is_search=True)
        n = _input.size(0)
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

        _input = _input.clone().detach().requires_grad_(False).cuda()
        _target = _target.clone().detach().requires_grad_(False).cuda(non_blocking=True)

        logits = model(_input)
        logits = utils.quantize(logits, args.rgb_range)

        loss = criterion(logits, _target)

        psnr = utils.calc_psnr(logits, _target, idx_scale,
                               args.rgb_range, is_search=True)
        n = _input.size(0)
        objs.update(loss.item(), n)
        eval_index.update(psnr, n)

        if step % args.report_freq == 0:
            logging.info('valid step: %03d   loss: %e    PSNR: %.2f', step,
                         objs.avg, eval_index.avg)

    return eval_index.avg, objs.avg


if __name__ == '__main__':
    main()
