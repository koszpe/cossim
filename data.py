import torch
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(args):
    fn = create_function(args.number_of_fn_part)
    datasets = {
        "train": FnDataset(fn=fn, in_len=args.in_len, start=args.train_start, stop=args.val_start, step=args.step),
        "val": FnDataset(fn=fn, in_len=args.in_len, start=args.val_start, stop=args.test_start, step=args.step),
        "test": FnDataset(fn=fn, in_len=args.in_len, start=args.test_start, stop=args.test_start * 2 - args.val_start, step=args.step)
    }

    dataloaders = {
        k: DataLoader(ds,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      pin_memory=True) for k, ds in datasets.items()
    }

    return dataloaders


def create_function(number_of_poly, range=(0, 100)):
    a = torch.FloatTensor(number_of_poly).uniform_(range[0], range[1]).T
    b = torch.FloatTensor(number_of_poly).uniform_(range[0], range[1]).T
    c = torch.FloatTensor(number_of_poly).uniform_(range[0], range[1]).T
    def function(t):
        return torch.sign(torch.sin((a * t**2 + b*t + c))).sum(dim=-1)
    return function


class FnDataset(Dataset):
    def __init__(self, fn, in_len, start, stop, step=1.0):
        super().__init__()
        self.fn = fn
        self.in_len = in_len
        self.start = start
        self.stop = stop
        self.step = step

    def __len__(self):
        return int((self.stop - self.start) // self.step - (self.in_len + 1))

    def __getitem__(self, idx):
        t = self.start + idx * self.step
        assert t < self.stop
        data = self.fn(torch.arange(t, t + self.step * (self.in_len + 1), self.step).unsqueeze(-1))
        return data[:-1], data[-1]
