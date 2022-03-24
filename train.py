import argparse
import os
from datetime import datetime

import numpy as np
import torch.cuda
from tqdm import tqdm

from data import get_dataloaders
from logger import TBLogger
from misc import ScaleGrad
from models import __configs__, MLP
from torch import nn, optim
import matplotlib.pyplot as plt


def train(model, dataloader, device, optimizer, similarity, epoch, tb_logger):
    log_per_step = 4096
    tqdm_iter = tqdm(dataloader, desc="training ? epoch. loss: ? accuracy: ?", leave=True, ncols=170)
    similarities, losses = [], []
    after_similarities, norms, cossim_diffs = [], [], []
    grads, grad_norms = [], []
    for input, target in tqdm_iter:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        output = output.squeeze()
        output.register_hook(lambda grad: grads.append(grad.mean(-1).cpu()))
        if args.scale_grad:
            output = ScaleGrad.apply(output)
            target = ScaleGrad.apply(target)
        cossim = similarity(output, target)
        loss = - cossim.mean()
        loss.backward()
        norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_norms.append(norm_before_clip)
        optimizer.step()
        with torch.no_grad():
            output = model(input)
            output = output.squeeze()
            after_cossim = similarity(output, target)
        similarities.append(cossim.mean())
        after_similarities.append(after_cossim.mean())

        cossim_diff = after_cossim - cossim
        target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
        output_norm = torch.norm(output, p=2, dim=1, keepdim=True)

        cossim_diffs.append(cossim_diff.mean().detach().cpu())
        # norms.append(((target_norm + output_norm) / 2).detach().cpu())
        norms.append(output_norm.detach().cpu())

        losses.append(loss.item())
        tqdm_iter.set_description(
            f"training - {epoch}. epoch loss: {torch.Tensor(losses).mean():.8f} cossim: {torch.Tensor(similarities).mean():.6f}"
            f" as cossim: {torch.Tensor(cossim_diffs).mean():.6f} o norm: {torch.cat(norms).mean():.3f} "
            f"o std: {torch.cat(norms).std():.4f}, grad norm: {torch.Tensor(grad_norms).mean():.3f}")
        tqdm_iter.refresh()
        if tb_logger.need_log(log_per_step):
            tb_logger.add_scalar(tag=f"train/loss", scalar_value=loss.item())
            tb_logger.add_scalar(tag=f"train/similarity", scalar_value=cossim.mean().item())
            tb_logger.add_scalar(tag=f"train/similarity_change_after_step", scalar_value=cossim_diff.mean().item())

            tb_logger.log_describe(name=f"train/output_norm", tensor=torch.cat(norms))
            tb_logger.log_describe(name=f"train/grad_norm", tensor=torch.Tensor(grad_norms))
        tb_logger.step()

    # plt.scatter(torch.cat(norms).squeeze(), torch.cat(cossim_diffs))
    # # plt.scatter(norms[-1].squeeze(), cossim_diffs[-1])
    # plt.title("x: norm, y: cossim diff")
    # # plt.yscale("log")
    # plt.show()
    #
    # plt.scatter(torch.cat(norms).squeeze(), torch.abs(torch.cat(grads)))
    # # plt.scatter(norms[-1].squeeze(), cossim_diffs[-1])
    # plt.title("x: norm, y: grads")
    # # plt.yscale("log")
    # plt.show()

def validate(model, dataloader, device, similarity, tb_logger, prefix="Validation"):
    tqdm_iter = tqdm(dataloader, desc=f"{prefix}...", leave=False)
    similarities, batch_sizes = [], []
    for input, target in tqdm_iter:
        input, target = input.to(device), target.to(device)
        output = model(input)
        output = output.squeeze()
        cossim = similarity(output, target).mean()
        similarities.append(cossim.item())
        batch_sizes.append(len(target))
    cossim = np.average(similarities, weights=batch_sizes)
    print(f"{prefix} cossim: {cossim:.6f}")

def main(args):
    dataloader = get_dataloaders(args)
    model_config = __configs__[args.config]
    ModelCLS = model_config.pop("type")
    if ModelCLS is MLP:
        model_config["in_size"] = args.vector_dim * 2
        model_config["out_size"] = args.vector_dim
    model = ModelCLS(**model_config)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)
    similarity = nn.CosineSimilarity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train on {device}.")
    model.to(device)
    runname = f"{args.runname}_vector_dim_{args.vector_dim}_op_{args.operation_type}" \
              f"_bs_{args.batch_size}_lr_{args.lr}_opt_{args.optimizer}" \
              f"_config_{args.config}_scalegrad_{args.scale_grad}_gradclip_{args.grad_clip}"
    runname += f"_{datetime.now().strftime('%Y-%m-%d_%H%M%S%f')}"
    folder = os.path.join(args.logdir, runname)
    os.makedirs(folder, exist_ok=True)
    global_step = len(dataloader['train'])
    tb_logger = TBLogger(log_dir=folder,
                         global_step=global_step,
                         batch_size=args.batch_size)
    for epoch in range(args.epochs):
        train(model, dataloader['train'], device, optimizer, similarity, epoch, tb_logger)
        if args.val_freq > 0 and epoch % args.val_freq == 0 :
            with torch.no_grad():
                validate(model, dataloader['val'], device, similarity, tb_logger)
    with torch.no_grad():
        validate(model, dataloader['test'], device, similarity, tb_logger, prefix="Test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Higher Order Fuorier Training')

    parser.add_argument("--runname", default="dev", help="Name of run on tensorboard")
    parser.add_argument('--logdir', default="/storage/cossim/logs/gradscale", type=str, help='Where to log')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--optimizer', help='optimizer for training', default='SGD')
    parser.add_argument('--val_freq', default=0, type=int, help='How often to evaluate')

    parser.add_argument('--vector-dim', default=128, type=int, help='number of polynom')
    parser.add_argument('--operation-type', default="sum", choices=["hadamard", "sum"], type=str, help='degree of polynom')

    parser.add_argument('--train-size', default=100000, type=int, help='train set first t')
    parser.add_argument('--val-size', default=50000, type=int, help='validation set first t')
    parser.add_argument('--test-size', default=50000, type=int, help='test set first t')

    parser.add_argument('--batch-size', default=1024, type=int, help='training batch size')
    parser.add_argument('--num-workers', default=15, type=int, help='number of dataloader workers')

    parser.add_argument('--config', help='model config', default='mlp_100_100_100')
    parser.add_argument('--scale-grad', default=0, type=int, choices=[0, 1])

    parser.add_argument('--grad-clip', default=1000.0, type=float)

    args = parser.parse_args()
    main(args)
