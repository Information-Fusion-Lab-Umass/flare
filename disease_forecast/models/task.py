import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_DX(nn.Module):
    def __init__(self):
        super(ANN_DX, self).__init__()
        self.fc1 = nn.Linear(713, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 3)
        self.dp = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp(nn.BatchNorm1d(250)(x))
        x = F.relu(self.fc2(x))
        x = self.dp(nn.BatchNorm1d(100)(x))
        x = F.relu(self.fc3(x))
        return x

    def loss(self, ypred, y):
        return nn.CrossEntropyLoss()(ypred, y)
