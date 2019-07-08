import torch
import utils
import data
import model
from loss import Loss
from model.model import Network
from configs.train_configs import args as args
from trainer import Trainer
from data import DataLoader

torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)

def main():
    if checkpoint.ok:
        data_loader = DataLoader(args)
        loss = Loss(args, checkpoint) if not args.test_only else None
        model = Network(args).cuda()
        srdarts = Trainer(args, data_loader, model, loss, checkpoint)

        while not srdarts.terminate():
            srdarts.train()
            srdarts.test()

        checkpoint.done()

if __name__ == '__main__':
    main()