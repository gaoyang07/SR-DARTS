from collections import namedtuple

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    "skip_connect",
    'conv_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

SRdarts_V1 = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1),
            ('conv_3x3', 2), ('conv_3x3', 2), ('sep_conv_3x3', 1),
            ('group_conv_3x3_2', 4), ('skip_connect', 1)],
    normal_concat=range(2, 6))

# PSNR: 37.552	SSIM: 0.9650
# Bad performance cause no skip-connect is included.
SRdarts_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('group_conv_3x3_4', 1), ('conv_3x3', 2),
            ('sep_conv_3x3', 0), ('group_conv_3x3_2', 3),
            ('group_conv_3x3_2', 2), ('group_conv_3x3_4',
                                      3), ('group_conv_3x3_2', 2)],
    normal_concat=range(2, 6))

# PSNR: 29.288  SSIM: 0.9390(Epoch 18)
SRdarts_V3 = Genotype(
    normal=[('group_conv_3x3_4', 0), ('skip_connect', 1), ('skip_connect', 1),
            ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('skip_connect', 0),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)],
    normal_concat=range(2, 6))
