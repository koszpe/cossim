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
        return x.squeeze()


__configs__ = {
    "mlp_10_10_10": {
        "type": MLP,
        "hidden_sizes": [10, 10, 10]
    },
    "mlp_100_100_100": {
        "type": MLP,
        "hidden_sizes": [100, 100, 100]
    }
}
