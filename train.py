import argparse

import torch.cuda
from tqdm import tqdm

from data import get_dataloaders
from models import __configs__
from torch import nn, optim


def main(args):
    dataloader = get_dataloaders(args)
    model_config = __configs__[args.config]
    ModelCLS = model_config.pop("type")
    model = ModelCLS(**model_config)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train on {device}.")
    model.to(device)
    for epoch in range(args.epochs):
        tqdm_iter = tqdm(dataloader['train'], desc="training ? epoch. loss: ?", leave=True)
        for input, target in tqdm_iter:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            tqdm_iter.set_description(f"training - {epoch}. epoch loss: {loss.item():.8f}")
            tqdm_iter.refresh()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Higher Order Fuorier Training')

    parser.add_argument("--runname", default="dev", help="Name of run on tensorboard")
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--optimizer', help='optimizer for training', default='AdamW')

    parser.add_argument('--number-of-fn-part', default=1, type=int, help='number of polynom')
    parser.add_argument('--p-degree', default=2, type=int, help='degree of polynom')

    parser.add_argument('--train-start', default=0, type=int, help='train set first t')
    parser.add_argument('--val-start', default=100000, type=int, help='validation set first t')
    parser.add_argument('--test-start', default=150000, type=int, help='test set first t')
    parser.add_argument('--step', default=1.0, type=float, help='step of t')
    parser.add_argument('--in-len', default=9, type=int, help='step of t')

    parser.add_argument('--batch-size', default=1024, type=int, help='training batch size')
    parser.add_argument('--num-workers', default=15, type=int, help='number of dataloader workers')

    parser.add_argument('--config', help='model config', default='mlp_10_10_10')

    args = parser.parse_args()
    main(args)
