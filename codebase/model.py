import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class CNN1(Module):
    def __init__(self):
        super(CNN1, self).__init__()

        # input: 15x15x3
        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 15x15x32
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 15x15x64
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        # 7x7x64
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.fc1 = Linear(7 * 7 * 64, 128)
        # 128x1
        self.fc2 = Linear(128, 3)
        # 3x1
        self.softmax = Softmax()

    # Defining the forward pass    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 7*7*64)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        return x