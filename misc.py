import torch.autograd

norm_fn = lambda x: torch.norm(x.mean(dim=0), p=2)

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
        grad_norm = norm_fn(grad_output)
        scaled_grad = grad_output * norm ** 2
        scaled_grad_norm = norm_fn(scaled_grad)
        rescaled_grad = scaled_grad * (grad_norm / scaled_grad_norm)
        # rescaled_grad = scaled_grad * (0.001 / scaled_grad_norm)

        # print(f"orig: {grad_mean.item()} scaled: {scaled_grad_mean.item()} ratio: {(grad_mean / scaled_grad_mean)} rescaled: {rescaled_grad.mean().item()}")
        return rescaled_grad
        # return grad_output * nor