import torch
import utils
import data
import model
import numpy as np
from loss import Loss
from data import DataLoader
from trainer import Trainer
from model.train.model import Network
from configs.train_configs import args


def main():
    """
    Main Function for training process.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint = utils.checkpoint(args)
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