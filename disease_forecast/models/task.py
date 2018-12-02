import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_DX(nn.Module):
    def __init__(self):
        super(ANN_DX, self).__init__()
        self.fc1 = nn.Linear(521, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 3)
        self.dp = nn.Dropout(p=0.5)

    #  def layer(x, n_in, n_out, p=0.5):
    #      x = nn.Linear(n_in, n_out)(x)
    #      x = F.relu(x)
    #      x = nn.BatchNorm1d(n_out)(x)
    #      x = nn.Dropout(p)(x)
    #      return x

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp(nn.BatchNorm1d(400)(x))
        x = F.relu(self.fc2(x))
        x = self.dp(nn.BatchNorm1d(200)(x))
        x = F.relu(self.fc3(x))
        x = self.dp(nn.BatchNorm1d(100)(x))
        x = F.relu(self.fc4(x))
        return x

    def loss(self, ypred, y):
        return nn.CrossEntropyLoss()(ypred, y)

