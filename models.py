from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size=9, out_size=1, hidden_sizes=[10, 10, 10], use_bn=False):
        super().__init__()
        self.linears = []
        self.bns = []
        self.layer_num = len(hidden_sizes)
        for i, (in_s, out_s)  in enumerate(zip([in_size] + hidden_sizes, hidden_sizes + [out_size])):
            linear = nn.Linear(in_s, out_s)
            setattr(self, f"linear_{i}", linear) # This line register the layer as parameters of the module
            self.linears.append(linear)
            if use_bn and i < self.layer_num:
                bn = nn.BatchNorm1d(out_s)
            else:
                bn = nn.Identity()
            setattr(self, f"bn_{i}", bn)
            self.bns.append(bn)

    def forward(self, x):
        for i, (l, bn) in enumerate(zip(self.linears, self.bns)):
            x = l(x)
            x = bn(x)
            if i < self.layer_num:
                x = F.relu(x)
        return x


class LSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=100, num_layers=1, dropout=0, out_size=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.hidden2logit = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(-1))
        logit = self.hidden2logit(lstm_out[:, -1, :])
        return logit

__configs__ = {
    # MLP configs
    "mlp_10_10_10": {
        "type": MLP,
        "hidden_sizes": [10, 10, 10]
    },
    "mlp_100_100_100": {
        "type": MLP,
        "hidden_sizes": [100, 100, 100]
    },
    "mlp_100_100_100_bn": {
        "type": MLP,
        "hidden_sizes": [100, 100, 100],
        "use_bn": True
    },

    # LSTM configs
    "lstm_100_2": {
        "type": LSTM,
        "hidden_size": 100,
        "num_layers": 2,
    },
    "lstm_100_2_do": {
        "type": LSTM,
        "hidden_size": 100,
        "num_layers": 2,
        "dropout": 0.5
    },
    "lstm_100_1": {
        "type": LSTM,
        "hidden_size": 100,
        "num_layers": 1,
    },
    "lstm_100_5_do": {
        "type": LSTM,
        "hidden_size": 100,
        "num_layers": 5,
        "dropout": 0.5
    },
    "lstm_100_5": {
        "type": LSTM,
        "hidden_size": 100,
        "num_layers": 5,
    },

}
