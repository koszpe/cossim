from torch import nn


class MLP(nn.Module):
    def __init__(self, in_size=9, out_size=1, hidden_sizes=[10, 10, 10]):
        super().__init__()
        self.linears = []
        for i, (in_s, out_s)  in enumerate(zip([in_size] + hidden_sizes, hidden_sizes + [out_size])):
            linear = nn.Linear(in_s, out_s)
            setattr(self, f"linear_{i}", nn.Linear(in_s, out_s))
            self.linears.append(linear)

    def forward(self, x):
        for l in self.linears:
            x = l(x)
        return x