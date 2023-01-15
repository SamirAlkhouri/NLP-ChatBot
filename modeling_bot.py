import torch
import torch.nn as nn


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, number_classes):
        super(NeuralNetworkModel, self).__init__()
        # input layers
        self.l1 = nn.Linear(input_size, hidden_size)
        # hidden layers
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # output layers
        self.l3 = nn.Linear(hidden_size, number_classes)
        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Passing input tensor through each operation
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
