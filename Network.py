import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(11520, 10)
        self.flatten = nn.Flatten()

        self.Relu = nn.ReLU()
        self.Softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.Relu(self.conv1(x))
        x = self.Relu(self.conv2(x))
        x = self.Softmax(self.fc1(self.flatten(x)))
        return x