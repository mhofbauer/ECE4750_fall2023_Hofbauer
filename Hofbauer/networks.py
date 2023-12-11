import torch
import torch.nn as nn

class AgeCNN(nn.Module):
    def __init__(self, output_dim: int):
        super(AgeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=(1, 1), padding=(0, 0))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.linear1 = nn.Linear(12544, output_dim)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
        x = self.relu3(self.linear1(x))
        return x
