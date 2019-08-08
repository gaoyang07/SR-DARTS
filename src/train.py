import torch
import data
import model
import random
import numpy as np
import utils.utils as utils
import torch.backends.cudnn as cudnn
from loss import Loss
from data import DataLoader
from engine.trainer import Trainer
from configs.train_configs import args
from model.train.model import Model as Model

def main():
    """
    Main Function for training process.
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

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