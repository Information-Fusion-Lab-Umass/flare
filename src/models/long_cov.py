import torch
import torch.nn as nn
import torch.nn.functional as F

class Longitudinal(nn.Module):
    def __init__(self):
        super(Longitudinal, self).__init__()
        self.affine = nn.Linear(4, 10)
        
    def forward(self, x): 
        x = self.affine(x)
        x = nn.BatchNorm1d(x.shape[1])(x)
        x = F.relu(x)
        return x

class Covariate(nn.Module):
    def __init__(self):
        super(Covariate, self).__init__()
        self.affine = nn.Linear(3, 10)
        
    def forward(self, x, output='tensor'):        
        x = self.affine(x)
        x = nn.BatchNorm1d(x.shape[1])(x)
        x = F.relu(x)
        return x


