from collections import namedtuple

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    "skip_connect",
    'conv_1x1',
    'conv_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'group_conv_3x3_2',
    'group_conv_3x3_4',
]

SRdarts_V1 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('conv_3x3', 2), (
    'conv_3x3', 2), ('sep_conv_3x3', 1), ('group_conv_3x3_2', 4), ('skip_connect', 1)], normal_concat=range(2, 6))
