import torch
import utils
import data
import model
from loss import Loss
from model.model_search import Network
from configs.search_configs import args as args
from searcher import Searcher
from data import DataLoader

torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)

def main():
    if checkpoint.ok:
        data_loader = DataLoader(args)
        loss = Loss(args, checkpoint) if not args.test_only else None
        model = Network(args, loss).cuda()
        srdarts = Searcher(args, data_loader, model, loss, checkpoint)

        while not srdarts.terminate():
            srdarts.search()
            srdarts.valid()

        checkpoint.done()

if __name__ == '__main__':
    main()
    