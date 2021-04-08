import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#10, 20, 10, 5

class ListNet(nn.Module):
    def __init__(self, D):
        super(ListNet, self).__init__()
        self.l1 = nn.Linear(D, 20)
        self.l2 = nn.Linear(20, 10)
        self.l3 = nn.Linear(10, 5)
        self.l4 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.l2(F.relu(self.l1(x)))
        x = torch.sigmoid(self.l3(x))
        x = self.l4(x)
        return x
    def predict(self, x):
        x = self.l2(F.relu(self.l1(x)))
        x = torch.sigmoid(self.l3(x))
        x = self.l4(x)
        return x