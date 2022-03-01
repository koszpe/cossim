import argparse

from data import get_dataloaders


def main(args):
    dataloader = get_dataloaders(args)
    print(dataloader)

    for input, target in dataloader['train']:
        print(input)
        print(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Higher Order Fuorier Training')

    parser.add_argument("--runname", default="dev", help="Name of run on tensorboard")
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--optimizer',
                        help='path to dataset', default='Adam')

    parser.add_argument('--number-of-poly', default=5, type=int,
                        help='number of polynoms')

    parser.add_argument('--train-start', default=0, type=int,
                        help='train set first t')
    parser.add_argument('--val-start', default=100000, type=int,
                        help='validation set first t')
    parser.add_argument('--test-start', default=150000, type=int,
                        help='test set first t')
    parser.add_argument('--step', default=1.0, type=float,
                        help='step of t')
    parser.add_argument('--in-len', default=9, type=int,
                        help='step of t')

    parser.add_argument('--batch-size', default=1024, type=int,
                        help='training batch size')
    parser.add_argument('--num-workers', default=15, type=int,
                        help='number of dataloader workers')


    args = parser.parse_args()
    main(args)
