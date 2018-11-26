import torch
import torch.nn as nn
import torch.nn.functional as F

class AppendTime(nn.Module):
    def __init__(self):
        super(AppendTime, self).__init__()
        
    def forward(self, x, t): 
        diff_t = (t[:,-1] - t[:,-2]).view(-1, 1)
        x = torch.cat((x, diff_t), 1) 
        return x


