import torch
import torch.nn as nn
import torch.nn.functional as F

class Longitudinal(nn.Module):
    def __init__(self):
        super(Longitudinal, self).__init__()
        self.affine = nn.Linear(4, 10)
        
    def forward(self, x):        
        x = F.relu(self.affine(x))
        return x

class Covariate(nn.Module):
    def __init__(self):
        super(Covariate, self).__init__()
        self.affine = nn.Linear(3, 10)
        
    def forward(self, x, output='tensor'):        
        x = F.relu(self.affine(x))
        return x


