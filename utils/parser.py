import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'Cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--logger', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--preconditioner', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--update_freq', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--hyperparam', type=str, default=None)
    return parser

args, unknown = get_args_parser().parse_known_args()