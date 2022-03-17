import torch.autograd


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # norm = torch.norm(x, p=2, dim=1, keepdim=True)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # return grad_output * norm ** 2 / norm.mean() ** 2
        # return grad_output * torch.sqrt(x ** 3) / torch.sqrt(x.mean() ** 3)
        # x = (x ** 2).sum(dim=-1, keepdim=True)
        # return grad_output * torch.sqrt(x ** 3) / torch.sqrt(x.mean() ** 3)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # print(grad_output.shape)
        # print((x @ grad_output.T).sum(-1))
        # for a, b in zip(x, grad_output):
        #     print(torch.dot(a, b))
        return grad_output * norm ** 2 / norm.mean() ** 2