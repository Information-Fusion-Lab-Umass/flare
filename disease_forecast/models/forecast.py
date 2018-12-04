import torch
import torch.nn as nn
import torch.nn.functional as F
from disease_forecast import utils

class AppendTime(nn.Module):
    def __init__(self):
        super(AppendTime, self).__init__()
        
    def forward(self, x, t):
        if t.shape[1] != 1:
            t = (t[:,-1] - t[:,-2]).view(-1, 1)
            diff_t_onehot = utils.one_hot(t)
            x = torch.cat((x, diff_t_onehot), 1) 
        return x

class MultiplyTime(nn.Module):
    def __init__(self):
        super(MultiplyTime, self).__init__()
        
    def forward(self, x, t): 
        if t.shape[1] != 1:
            diff_t = (t[:,-1] - t[:,-2]).view(-1, 1)
            x = x*diff_t 
        return x


