import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_DX(nn.Module):
    def __init__(self, num_input, num_classes):
        super(ANN_DX, self).__init__()
        self.fc1 = nn.Linear(num_input, 450)
        self.bn1 = nn.BatchNorm1d(450)

        self.fc2 = nn.Linear(450, 450)
        self.bn2 = nn.BatchNorm1d(450)

        self.fc3 = nn.Linear(450, 450)
        self.bn3 = nn.BatchNorm1d(450)

        self.fc4 = nn.Linear(450, 450)
        self.bn4 = nn.BatchNorm1d(450)

        self.fc5 = nn.Linear(450, 200)
        self.bn5 = nn.BatchNorm1d(200)

        self.fc6 = nn.Linear(200, 100)
        self.bn6 = nn.BatchNorm1d(100)

        self.fc7 = nn.Linear(100, num_classes)

        self.dp1 = nn.Dropout(p=0.0)
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x):
        def layer(x, fc, dp, bn):
            return dp(bn(F.relu(fc(x))))
        
        x = layer(x, self.fc1, self.dp1, self.bn1)
        x = layer(x, self.fc2, self.dp2, self.bn2)
        x = layer(x, self.fc3, self.dp2, self.bn3)
        x = layer(x, self.fc4, self.dp2, self.bn4)
        x = layer(x, self.fc5, self.dp1, self.bn5)
        x = layer(x, self.fc6, self.dp1, self.bn6)
        x = self.fc7(x)

        return x

class ANN_DX_1(nn.Module):
    def __init__(self, num_input, num_classes):
        super(ANN_DX_1, self).__init__()
        self.fc1 = nn.Linear(num_input, 1000)
        self.bn1 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, 1000)
        self.bn3 = nn.BatchNorm1d(1000)

        self.fc4 = nn.Linear(1000, 500)
        self.bn4 = nn.BatchNorm1d(500)

        self.fc5 = nn.Linear(500, 200)
        self.bn5 = nn.BatchNorm1d(200)

        self.fc6 = nn.Linear(200, 100)
        self.bn6 = nn.BatchNorm1d(100)

        self.fc7 = nn.Linear(100, num_classes)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.2)
        self.dp3 = nn.Dropout(p=0.1)

    def forward(self, x):
        def layer(x, fc, dp, bn):
            return dp(bn(F.relu(fc(x))))
        
        x = layer(x, self.fc1, self.dp1, self.bn1)
        x = layer(x, self.fc2, self.dp2, self.bn2)
        x = layer(x, self.fc3, self.dp2, self.bn3)
        x = layer(x, self.fc4, self.dp3, self.bn4)
        x = layer(x, self.fc5, self.dp3, self.bn5)
        x = layer(x, self.fc6, self.dp3, self.bn6)
        x = self.fc7(x)

        return x
