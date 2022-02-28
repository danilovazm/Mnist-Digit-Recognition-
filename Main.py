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
                            
lr = 0.0001
Net = cnn()
epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_losses, Net = Train(epochs, lr, Net, train_set, device)
plt.plot(train_losses, color='blue')
plotChart()
ax = Test(Net, test_set)
plotChart()