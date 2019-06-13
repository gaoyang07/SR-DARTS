import argparse


parser = argparse.ArgumentParser("SR-DARTS")
parser.add_argument('--dir_data', type=str, default='../datasets',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='location of the training data')
parser.add_argument('--data_valid', type=str, default='DIV2K',
                    help='location of the training data')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='location of the testing data')
parser.add_argument('--data_range', type=str, default='1-800/801-900',
                    help='train/test data range')

parser.add_argument('--search', action="store_true", default=True,
                    help='the searching process.')

parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--model_path', type=str, default='',
                    help='path to save the model')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='choose to keep training using the former model')
parser.add_argument('--save', type=str, default='output/test/',
                    help='experiment name')
parser.add_argument('--note', type=str, default='try',
                    help='note for this run')

parser.add_argument('--gpu', type=str, default='0',
                    help="gpu device id")
parser.add_argument('--n_GPUs', type=int, default=4,
                    help='number of GPUs')
parser.add_argument('--n_threads', type=int, default=1,
                    help='number of threads for data loading')

parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')

parser.add_argument('--epochs', type=int, default=100,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')

parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--learning_rate', type=float,
                    default=5e-3, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=1e-4, help='min learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for the optimizer')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4,
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                    help='weight decay for arch encoding')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--init_channels', type=int,
                    default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')

parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--init_channels', type=int,
                    default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true',
                    default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SRdarts_V1',
                    help='which architecture to use')

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_valid = args.data_valid.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8