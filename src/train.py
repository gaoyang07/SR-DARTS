import torch
import data
import model
import numpy as np
from loss import Loss
import utils.utils as utils
from data import DataLoader
from engine.trainer import Trainer
from configs.train_configs import args
from model.train.model import Model as Model

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
        train_model = Model(args, loss, checkpoint)
        srdarts = Trainer(args, data_loader, train_model, loss, checkpoint)

        while not srdarts.terminate():
            srdarts.train()
            srdarts.test()

        checkpoint.done()

if __name__ == '__main__':
    main()