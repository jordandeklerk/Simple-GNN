import torch 
import random
import numpy as np
from utils import *
from parser import get_args_parser
from dataloader import get_planetoid_dataset
from model import GNN
from train import Trainer


def main():
    args, unknown = get_args_parser().parse_known_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("\n--- GPU Information ---\n")

    if torch.cuda.is_available():
        print(f"Model is using device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2} MB")
    else:
        print("Model is using CPU")

    print("\n--- Downloading Data ---\n")

    if __name__ == '__main__':
        lst_names = ['Cora', 'CiteSeer', 'PubMed']
        for name in lst_names:
            dataset = get_planetoid_dataset(name)
            print(f"dataset: {name}")
            print(f"num_nodes: {dataset[0]['x'].shape[0]}")
            print(f"num_edges: {dataset[0]['edge_index'].shape[1]}")
            print(f"num_classes: {dataset.num_classes}")
            print(f"num_features: {dataset.num_node_features}")

    dataset = get_planetoid_dataset(name=args.dataset, normalize_features=args.normalize_features, split=args.split)

    print("\n--- Training Model ---\n")

    kwargs = {
    'dataset': dataset,
    'model': GNN(dataset),
    'str_optimizer': args.optimizer,
    'runs': args.runs,
    'epochs': args.epochs,
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'early_stopping': args.early_stopping,
    'logger': args.logger,
    'momentum': args.momentum,
    'eps': args.eps,
    'update_freq': args.update_freq,
    'gamma': args.gamma,
    'alpha': args.alpha,
    'hyperparam': args.hyperparam
    }

    trainer = Trainer(**kwargs)

    if args.hyperparam == 'eps':
      for param in np.logspace(-3, 0, 10, endpoint=True):
        print(f"{args.hyperparam}: {param}")
        kwargs[args.hyperparam] = param
        trainer.train()
    elif args.hyperparam == 'update_freq':
      for param in [4, 8, 16, 32, 64, 128]:
        print(f"{args.hyperparam}: {param}")
        kwargs[args.hyperparam] = param
        trainer.train()
    elif args.hyperparam == 'gamma':
      for param in np.linspace(1., 10., 10, endpoint=True):
        print(f"{args.hyperparam}: {param}")
        kwargs[args.hyperparam] = param
        trainer.train()
    else:
      trainer.train()

if __name__ == "__main__":
    main()