import os
import sys
import pdb
import utils
import torch
import torch.nn as nn

from decimal import Decimal
from model.common import *
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter


class Trainer():
    """
    Define a class called Trainer for training purpose.
    """

    def __init__(self, args, loader, model, loss, ckp):
        self.args = args
        self.ckp = ckp
        self.scale = args.scale
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.optimizer = utils.make_optimizer(args, self.model)
        self.scheduler = utils.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '\n\n[Epoch {}]\tLearning rate: {:.2e}'.format(
                epoch,
                Decimal(lr)
            )
        )
        self.loss.start_log()
        self.model.train()

        # eval_psnr = 0
        # eval_ssim = 0
        timer_data, timer_model = utils.timer(), utils.timer()
        for batch, (_input, _target, _, idx_scale) in enumerate(self.loader_train):
            _input, _target = self.prepare(_input, _target)
            timer_data.hold()
            timer_model.tic()

            # TODO: check whether it's ness or not
            _input = _input.clone().detach().requires_grad_(False).cuda()
            _target = _target.clone().detach().requires_grad_(False).cuda(non_blocking=True)

            self.optimizer.zero_grad()

            # output is the high-resolution image
            logits = self.model(_input)
            loss = self.loss(logits, _target)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.grad_clip)
            self.optimizer.step()
            timer_model.hold()

            # self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_train.dataset)
            
            # best = self.ckp.log.max(0)
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}\t{}\t{:.1f}+{:.1f}s]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    # eval_psnr / len(self.loader_train.dataset),
                    # eval_ssim / len(self.loader_train.dataset),
                    # best[0][idx_scale],
                    # best[1][idx_scale] + 1,
                    timer_model.release(),
                    timer_data.release()
                ))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir,'model', 'model_{}.pt'.format(epoch))
        )

    def test(self):
        torch.set_grad_enabled(False)
        
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation: ')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        timer_test = utils.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                eval_acc = 0
                eval_acc_ssim = 0
                for _input, _target, filename, _ in tqdm(d, ncols=80):
                    scale = self.args.scale[idx_scale]
                    filename = filename[0]
                    
                    _input, _target = self.prepare(_input, _target)
                    timer_test.tic()

                    logits = self.model(_input)
                    timer_test.hold()
                    logits = utils.quantize(logits, self.args.rgb_range)
                    
                    save_list = [logits]
                    self.ckp.log[-1, idx_data, idx_scale] += utils.calc_psnr(
                        logits, _target, scale, self.args.rgb_range,
                        benchmark=d.dataset.benchmark
                    )
                    eval_acc_ssim += utils.calc_ssim(
                        logits, _target, scale,
                        benchmark=d.dataset.benchmark
                    )
                    save_list.extend([_input, _target])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '\n[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        eval_acc_ssim / len(d),
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
