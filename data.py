import torch
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(args):
    fn = create_function(args.number_of_fn_part, args.p_degree)
    datasets = {
        "train": FnDataset(fn=fn, in_len=args.in_len, size=args.train_size),
        "val": FnDataset(fn=fn, in_len=args.in_len, size=args.val_size),
        "test": FnDataset(fn=fn, in_len=args.in_len, size=args.test_size)
    }

    dataloaders = {
        k: DataLoader(ds,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      pin_memory=True) for k, ds in datasets.items()
    }

    return dataloaders


def create_function(number_of_poly, p_degree, coef_range=(0, 100)):
    p_degree += 1
    C = torch.FloatTensor(p_degree, number_of_poly).uniform_(coef_range[0], coef_range[1])
    print(f"generator C: :\n{C.numpy()}")
    def function(t):
        t_pow = torch.cat([t ** i for i in reversed(range(p_degree))], dim=-1)
        P_t = t_pow @ C
        return (torch.sign(torch.sin((P_t)).sum(dim=-1)) + 1) / 2
    return function


class FnDataset(Dataset):
    def __init__(self, fn, in_len, size):
        super().__init__()
        self.fn = fn
        self.in_len = in_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.fn(torch.arange(self.in_len + 1, dtype=torch.float32).unsqueeze(-1))
        return data[:-1] - 0.5, data[-1]
