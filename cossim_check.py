import argparse

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_factor):
        ctx.save_for_backward(x)
        ctx.scale_factor = scale_factor
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        # return grad_output * norm ** 2 / norm.mean() ** 2
        # return grad_output * torch.sqrt(x ** 3) / torch.sqrt(x.mean() ** 3)
        # x = (x ** 2).sum(dim=-1, keepdim=True)
        # return grad_output * torch.sqrt(x ** 3) / torch.sqrt(x.mean() ** 3)
        norm = torch.norm(x, p=2, dim=0, keepdim=True)
        # print(grad_output.shape)
        # print((x @ grad_output.T).sum(-1))
        # for a, b in zip(x, grad_output):
        #     print(torch.dot(a, b))
        return grad_output * norm ** scale_factor, None


def scatter(x, y, title='', x_label='', y_label=''):
    plt.scatter(x, y)
    plt.title(title, y=1.0, pad=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def get_grads(a, b, scale_factor):
    a, b = a.clone(), b.clone()
    a_s, b_s, a_norms, b_norms, a_grads, b_grads, cossims, cossim_diffs = [], [], [], [], [], [], [], []
    sim = nn.CosineSimilarity(dim=0)
    for i in tqdm(range(args.sample_number)):
        rand_coef = (i + 1) / args.sample_number * 10
        # rand_coef = torch.randint(low=1, high=10, size=[1])
        a_ = a * rand_coef
        b_ = b
        a_.requires_grad = True
        b_.requires_grad = True

        optimizer = torch.optim.SGD([a_], lr=0.01)
        optimizer.zero_grad()
        a_scaled = ScaleGrad.apply(a_, scale_factor)
        a_s.append(a_)
        b_s.append(b_)
        a_norms.append(torch.norm(a_, p=2, dim=0, keepdim=True).detach())
        b_norms.append(torch.norm(b_, p=2, dim=0, keepdim=True).detach())

        cossim = sim(a_scaled, b_)
        cossims.append(cossim.item())
        cossim.backward()
        a_grads.append(a_.grad.detach())
        b_grads.append(b_.grad.detach())

        cossim_old = cossim
        optimizer.step()
        with torch.no_grad():
            cossim = sim(a_scaled, b_)
        cossim_diffs.append((cossim_old - cossim).detach())

    return a_norms, b_norms, a_grads, b_grads, cossim_diffs

def main(args):
    a = torch.DoubleTensor(args.vector_dim).uniform_(-1, 1)
    b = torch.DoubleTensor(args.vector_dim).uniform_(-1, 1)

    a_norms, b_norms, a_grads, b_grads, cossim_diffs = get_grads(a, b, scale_factor=args.scale_grad)

    scatter(torch.cat(a_norms), torch.stack(a_grads).abs().mean(-1),
            title=f"Gradients of cossim with respect to a, scaling: grad * norm^{args.scale_grad}",
            x_label="norm of a",
            y_label="avg grad of a")
    scatter(torch.cat(a_norms), torch.stack(cossim_diffs),
            title=f"Cossim diff between before and after optim step, scaling: grad * norm^{args.scale_grad}",
            x_label="norm of a",
            y_label="cossim diff")
    # scatter(torch.cat(a_norms), torch.stack(b_grads).abs().mean(-1),
    #         title=f"Gradients of cossim with respect to b, scaling={args.scale_grad}",
    #         x_label="norm of a",
    #         y_label="avg grad of b")
    pass
    grads = torch.stack(a_grads)
    norm = torch.cat(a_norms).unsqueeze(-1).repeat(1, 128)
    grads.mean(), (grads * norm ** 2 / (norm ** 2).mean()).mean(), (grads * norm ** 2 / (norm).mean() ** 2).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Higher Order Fuorier Training')

    parser.add_argument('--vector-dim', default=128, type=int)
    parser.add_argument('--sample-number', default=100, type=int)

    parser.add_argument('--scale-grad', default=0, type=int)

    args = parser.parse_args()
    main(args)