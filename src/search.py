import torch
import utils
import data
import model
import numpy as np
from loss import Loss
from data import DataLoader
from searcher import Searcher
from configs.search_configs import args
from model.search import NetworkController as Model


def main():
    """
    Main Function for searching process.
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
        search_model = Model(args, loss).cuda()
        srdarts = Searcher(args, data_loader, search_model, loss, checkpoint)

        while not srdarts.terminate():
            srdarts.search()
            srdarts.valid()

        checkpoint.done()

if __name__ == '__main__':
    main()
    