import argparse
import torch
import torchvision
from Network import cnn
from train import Train
from test import Test
import matplotlib.pyplot as plt

def plotChart():
    plt.show()
    return

train_set = torch.utils.data.DataLoader( torchvision.datasets.MNIST('/dataset/', train=True, download=True, transform=torchvision.transforms.ToTensor()), batch_size=32, shuffle=True)

test_set = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/dataset/', train=False, download=True, transform=torchvision.transforms.ToTensor()))

def main(args):
    Net = cnn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses, Net = Train(args.epochs, args.lr, Net, train_set, device)
    plt.plot(train_losses, color='blue')
    plotChart()
    ax = Test(Net, test_set)
    plotChart()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    main(args)