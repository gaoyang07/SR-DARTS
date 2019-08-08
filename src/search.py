import torch
import data
import model
import random
import numpy as np
import utils.utils as utils
import torch.backends.cudnn as cudnn
from loss import Loss
from data import DataLoader
from engine.searcher import Searcher
from configs.search_configs import args
from model.search.controller import NetworkController as Controller


def main():
    """
    Main Function for searching process.
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
        search_model = Controller(args, loss).cuda()
        srdarts = Searcher(args, data_loader, search_model, loss, checkpoint)

        while not srdarts.terminate():
            srdarts.search()
            srdarts.valid()

        checkpoint.done()

if __name__ == '__main__':
    main()
    