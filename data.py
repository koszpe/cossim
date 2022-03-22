import torch
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(args):
    fn = create_function(args.vector_dim, args.operation_type)
    datasets = {
        "train": FnDataset(fn=fn, size=args.train_size),
        "val": FnDataset(fn=fn, size=args.val_size),
        "test": FnDataset(fn=fn, size=args.test_size)
    }

    dataloaders = {
        k: DataLoader(ds,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      pin_memory=True) for k, ds in datasets.items()
    }

    return dataloaders


def create_function(vector_dim, operation_type, coo_range=(-1, 1)):
    def function():
        rand_coef = torch.randint(low=1, high=10, size=[1])
        a = torch.FloatTensor(vector_dim).uniform_(coo_range[0], coo_range[1]) * rand_coef
        b = torch.FloatTensor(vector_dim).uniform_(coo_range[0], coo_range[1]) * rand_coef
        # b = a
        # b[0] = -a[1]
        # b[1] = a[0]
        if operation_type == "hadamard":
            target = a * b
        elif operation_type == "sum":
            target = a + b
        else:
            raise NotImplementedError
        return a, b, target
    return function


class FnDataset(Dataset):
    def __init__(self, fn, size):
        super().__init__()
        self.fn = fn
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        a, b, target = self.fn()
        return torch.cat([a, b]), target
