from collections import namedtuple

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype = namedtuple('Genotype', 'normal normal_concat')


PRIMITIVES = [
    'none',
    'conv_1x1',
    'conv_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'group_conv_3x3_2',
    'group_conv_3x3_4',
]

SRdarts_V1 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('conv_1x1', 1), (
    'group_conv_3x3_4', 2), ('group_conv_3x3_2', 3), ('group_conv_3x3_2', 4), ('group_conv_3x3_2', 3)], normal_concat=range(2, 6))
