from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
# ]
PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'sep_conv_3x3',
    'dil_conv_3x3',
    'group_conv_3x3_2',
    'group_conv_3x3_4',
]


NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[
                    2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[
                    2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS_V3 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(
    2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
DARTS_18 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(
    2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

DARTS_18_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(
    2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

DARTS_V6 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(
    2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

# DARTS_V7 acc 80.404 using search strategy 1
DARTS_V7 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('max_pool_3x3', 3), ('sep_conv_5x5', 4)], normal_concat=range(
    2, 6), reduce=[('skip_connect', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

Layer16 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 4), ('dil_conv_3x3', 2)], normal_concat=range(
    2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

DARTS_layer12 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 3), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2)], normal_concat=range(
    2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))

DARTS_new_search1 = Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 3), ('dil_conv_3x3', 3), ('avg_pool_3x3', 0)], normal_concat=range(
    2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 4), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))

DARTS_new_search2 = Genotype(normal=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 2)], normal_concat=range(
    2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 4), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))

DARTS_ASAP_50_epoch = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], normal_concat=range(
    2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

DARTS = DARTS_ASAP_50_epoch
