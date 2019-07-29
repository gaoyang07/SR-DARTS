from importlib import import_module
from data.dataloader import MSDataLoader
from torch.utils.data import ConcatDataset


class MyConcatDataset(ConcatDataset):

    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'):
                d.set_scale(idx_scale)


class DataLoader(object):

    def __init__(self, args):
        self.loader_train = None
        self.loader_valid = None
        self.loader_test = []

        if not args.test_only:
            datasets = []
            datasets_valid = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d, train=True))

            self.loader_train = MSDataLoader(
                args,
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
            )

            # data_valid shouldn't be used during training process.
            for d in args.data_valid:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets_valid.append(getattr(m, module_name)(args, name=d, train=False))

            self.loader_valid = MSDataLoader(
                args,
                MyConcatDataset(datasets_valid),
                batch_size=1,
                shuffle=False,
                pin_memory=True,
            )

        # data_test shouldn't be used during searching process.
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(MSDataLoader(
                args,
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
            ))
