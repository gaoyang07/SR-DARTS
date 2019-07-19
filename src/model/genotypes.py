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

SRdarts_V1 = Genotype(normal=[('conv_1x1', 0), ('dil_conv_3x3', 1), ('conv_1x1', 0), ('dil_conv_3x3', 1), (
    'conv_1x1', 0), ('dil_conv_3x3', 1), ('conv_1x1', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6))
