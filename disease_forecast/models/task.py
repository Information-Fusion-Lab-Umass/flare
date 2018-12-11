import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_DX(nn.Module):
    def __init__(self, num_input):
        super(ANN_DX, self).__init__()
        self.fc1 = nn.Linear(num_input, 400)
        self.bn1 = nn.BatchNorm1d(400)

        self.fc2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(400)

        self.fc3 = nn.Linear(400, 400)
        self.bn3 = nn.BatchNorm1d(400)

        self.fc4 = nn.Linear(400, 100)
        self.bn4 = nn.BatchNorm1d(100)

        self.fc5 = nn.Linear(100, 3)

        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        def layer(x, fc, dp, bn):
            x = F.relu(fc(x))
            return dp(bn(x))
        
        x = layer(x, self.fc1, self.dp1, self.bn1)
        x = layer(x, self.fc2, self.dp2, self.bn2)
        x = layer(x, self.fc3, self.dp2, self.bn3)
        x = layer(x, self.fc4, self.dp1, self.bn4)
        x = self.fc5(x)

        return x

    #  def __init__(self):
    #      super(ANN_DX, self).__init__()
    #      self.fc1 = nn.Linear(220, 100)
    #      #  self.fc2 = nn.Linear(200, 100)
    #      #  self.fc3 = nn.Linear(50, 10)
    #      self.fc4 = nn.Linear(100, 3)
    #      self.dp = nn.Dropout(p=0.2)
    #
    #  def forward(self, x):
    #      x = F.relu(self.fc1(x))
    #      x = self.dp(nn.BatchNorm1d(100)(x))
    #      #  x = F.relu(self.fc2(x))
    #      #  x = self.dp(nn.BatchNorm1d(100)(x))
    #      #  x = F.relu(self.fc3(x))
    #      #  x = self.dp(nn.BatchNorm1d(10)(x))
    #      x = F.relu(self.fc4(x))
    #      return x


