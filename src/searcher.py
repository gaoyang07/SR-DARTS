import pdb
import math
import torch
import utils
import torch.nn as nn
import torch.nn.functional as F

from decimal import Decimal
from model.common import *
from model.search.architect import Architect


class Searcher():
    """
    Define a class called Searcher for searching purpose
    """
    def __init__(self, args, loader, model, loss, ckp):
        self.args = args
        self.ckp = ckp
        self.scale = args.scale
        self.loader_search = loader.loader_train
        self.loader_valid = loader.loader_valid
        self.model = model
        self.loss = loss
        self.architect = Architect(self.model, self.args)
        self.optimizer = utils.make_optimizer(args, self.model)
        self.scheduler = utils.make_scheduler(args, self.optimizer)

        self.error_last = 1e8

    def search(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.visual("lr", lr, epoch)

        temperature = self.model.temperature
        self.ckp.visual("temp", temperature, epoch)
        self.model.temp_update(epoch)

        genotype = self.model.genotype()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}\tTemperature: {:.4e}\nGenotype: {}'.format(
                epoch,
                Decimal(lr),
                temperature,
                genotype
            )
        )
        print(F.softmax(self.model.alphas_normal, dim=-1))
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utils.timer(), utils.timer()
        for batch, (_input, _target, _, _) in enumerate(self.loader_search):
            _input, _target = self.prepare(_input, _target)
            timer_data.hold()
            timer_model.tic()
            # _input = _input.clone().detach().requires_grad_(False).cuda()
            # _target = _target.clone().detach().requires_grad_(False).cuda(non_blocking=True)

            input_search, target_search, _, _ = next(iter(self.loader_valid))
            input_search, target_search = self.prepare(input_search, target_search)

            # input_search = input_search.clone().detach().requires_grad_(False).cuda()
            # target_search = target_search.clone().detach(
            # ).requires_grad_(False).cuda(non_blocking=True)

            self.architect.step(_input, _target, input_search, target_search,
                                lr, self.optimizer, unrolled=self.args.unrolled)

            self.optimizer.zero_grad()
            logits = self.model(_input)
            loss = self.loss(logits, _target)
            loss.backward(retain_graph=True)

            # TODO:check the diff between clip_grad_norm and clip_grad_value_(in EDSR)
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.grad_clip)
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}\t{}\t{:.1f}+{:.1f}s]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_search.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()
                ))

            timer_data.tic()

        final_loss = self.loss.end_log(len(self.loader_search))
        final_loss = final_loss.numpy()[-1]
        self.ckp.visual("train_loss", final_loss, epoch)

        self.error_last = self.loss.log[-1, -1]

    def valid(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\n\nEvaluation during search process:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_valid = utils.timer()
        with torch.no_grad():
            eval_psnr = 0
            # eval_ssim = 0
            for batch, (_input, _target, _, idx_scale) in enumerate(self.loader_valid):
                _input, _target = self.prepare(_input, _target)

                # # TODO: check whether it's ness or not
                # _input = _input.clone().detach().requires_grad_(False).cuda()
                # _target = _target.clone().detach().requires_grad_(False).cuda(non_blocking=True)

                timer_valid.tic()
                logits = self.model(_input)
                timer_valid.hold()
                logits = utils.quantize(logits, self.args.rgb_range)

                eval_psnr += utils.calc_psnr(
                    logits, _target, self.scale[idx_scale], self.args.rgb_range,
                    benchmark=False
                )
                # eval_ssim += utils.calc_ssim(
                #     logits, _target, self.scale[idx_scale],
                #     benchmark=False
                # )

            self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_valid)

            best = self.ckp.log.max(0)
            # self.ckp.write_log(
            #     '[{} x{}]\tPSNR: {:.3f}\tSSIM: {:.4f}\t(best: {:.3f} @epoch {})'.format(
            #         self.args.data_valid,
            #         self.scale[idx_scale],
            #         self.ckp.log[-1, idx_scale],
            #         eval_ssim / len(self.loader_valid),
            #         best[0][idx_scale],
            #         best[1][idx_scale] + 1
            #     )
            # )
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f}\t(best: {:.3f} @epoch {})'.format(
                    self.args.data_valid,
                    self.scale[idx_scale],
                    self.ckp.log[-1, idx_scale],
                    best[0][idx_scale],
                    best[1][idx_scale] + 1
                )
            )
            self.ckp.visual("valid_PSNR", self.ckp.log[-1, idx_scale], epoch)
            # self.ckp.visual("valid_SSIM", eval_ssim /
            #                 len(self.loader_valid), epoch)

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_valid.toc()), refresh=True
        )

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.valid()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
