import argparse
from utils.train_utils import add_flags_from_config

# some generic hyper-parameters which may be used in every hyperbolic and Euclidean based KGC variant
config_args = {
    'training_config': {
        'lr': (0.005, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (500, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('radam', 'which optimizer to use, can be any of [rsgd, radam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'early-stop': (10, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'valid-steps': (5, 'how often to compute val metrics (in epochs)'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'), # could be used in lr_scheduler
        'gamma': (0.5, 'gamma for lr scheduler'), # could be used in lr_scheduler
        'grad-clip': (0.5, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs') # could be used for hyper-parameter tuning
    },
    'model_config': {
        'task': ('lp', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('HyboNet', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN, HyboNet]'),
        'dim': (16, 'embedding dimension'),
        'manifold': ('Lorentz', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall, Lorentz]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'), # currently c is fixed to 1 (curvature = -1) for every Lorentz Linear layer in the model
        'margin': (8., 'margin is used to control the margin hyper-parameter in KG completion score function'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('None', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'), # set the default torch tensor type to torch.float64
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'data': ('DTINet', 'which dataset to use'),
        'normalize-feats': (1, 'whether to normalize input node features'), # could be used later
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)

# add extra hyper-parameter for the KGC task
parser.add_argument('--real-neg', action='store_true', help='whether to use real unknown samples in KG completion training instead of random generated ones')
parser.add_argument('--GCN-add-layers', type=int, default=1, help='specify how many GCN_add layers will be used to fuse ontology and protein embeddings')
parser.add_argument('--max-norm', type=float, default=0, help='used in the lorentz projx/lmath project function to further control the norm of fused target embeddings (if exists)')

