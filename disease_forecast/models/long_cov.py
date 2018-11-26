import torch
import torch.nn as nn
import torch.nn.functional as F

class Longitudinal(nn.Module):
    def __init__(self):
        super(Longitudinal, self).__init__()
        self.affine = nn.Linear(4, 10)
        
    def forward(self, x, output='tensor'):        
        x = torch.from_numpy(x).float()
        x = F.relu(self.affine(x))
        if output=='numpy':
            x = x.data.numpy()
        return x

class Covariate(nn.Module):
    def __init__(self):
        super(Covariate, self).__init__()
        self.affine = nn.Linear(3, 10)
        
    def forward(self, x, output='tensor'):        
        x = torch.from_numpy(x).float()
        x = F.relu(self.affine(x))
        if output=='numpy':
            x = x.data.numpy()
        return x


