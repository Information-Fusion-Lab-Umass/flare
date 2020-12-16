import torch
import torch.nn as nn
import torch.nn.functional as F
from src import utils
import ipdb

class AppendZeros(nn.Module):
    def __init__(self, device):
        super(AppendZeros, self).__init__()
        self.device = device
        
    def forward(self, x, t):
        B = x.shape[0]
        t = t.view(-1, 1)
        zeros = torch.zeros(B,5).to(self.device)
        x = torch.cat((x, zeros), 1)

        return x

class AppendTime(nn.Module):
    def __init__(self, device):
        super(AppendTime, self).__init__()
        self.device = device
        
    def forward(self, x, t):
        t = t.view(-1, 1)
        diff_t_onehot = utils.one_hot(t, self.device)
        print(t.shape)
        print(diff_t_onehot.shape)
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

class AutoEncoder(nn.Module):
    def __init__(self, num_input, num_output):
        super(AutoEncoder, self).__init__()

        self.fc1 = nn.Linear(num_input, 250)
        self.bn1 = nn.BatchNorm1d(250)

        self.fc2 = nn.Linear(250, 250)
        self.bn2 = nn.BatchNorm1d(250)

        self.fc3 = nn.Linear(250, 250)
        self.bn3 = nn.BatchNorm1d(250)

        self.fc4 = nn.Linear(250, num_output)
        self.bn4 = nn.BatchNorm1d(num_output)

        self.dp1 = nn.Dropout(p=0.1)
        self.dp2 = nn.Dropout(p=0.5)
        self.dp3 = nn.Dropout(p=0.7)
        
    def forward(self, x, t):
        
        def layer(x, fc, dp, bn):
            return dp(bn(F.relu(fc(x))))
        
        if t.shape[1] != 1:
            t = (t[:,-1] - t[:,-2]).view(-1, 1)
            diff_t_onehot = utils.one_hot(t)
            x = torch.cat((x, diff_t_onehot), 1)

            x = layer(x, self.fc1, self.dp1, self.bn1)
            x = layer(x, self.fc2, self.dp2, self.bn2)
            x = layer(x, self.fc3, self.dp2, self.bn3)
            x = layer(x, self.fc4, self.dp3, self.bn4)
        return x

class Expand(nn.Module):
    def __init__(self, device, num_input):
        super(Expand, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_input, 12)
        self.bn1 = nn.BatchNorm1d(12)
        self.dp1 = nn.Dropout(p=0.1)

    def forward(self, x, t):
        def layer(x, fc, dp, bn):
            return dp(bn(F.relu(fc(x))))
        
        x = layer(x, self.fc1, self.dp1, self.bn1)
        return x

class T1(nn.Module):
    ''' 
    Completely different model for T=1 case.
    '''

    def __init__(self, device, num_input, num_timesteps):
        super(T1, self).__init__()
        self.device = device
        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 2,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)
        self.concat = AppendTime(device)

        self.fc1 = nn.Linear(12, 5)
        self.bn1 = nn.BatchNorm1d(5)
        
        self.fc2 = nn.Linear(5, 5)
        self.bn2 = nn.BatchNorm1d(5)

        self.fc3 = nn.Linear(5, 3)

        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x, t):
        def layer(x, fc, dp, bn):
            return dp(bn(F.relu(fc(x))))
        x = self.rnn(x)[0] 
        x = x[:,-1,:]
        x = self.concat(x,t)
        x = layer(x, self.fc1, self.dp1, self.bn1)
        x = layer(x, self.fc2, self.dp2, self.bn2)
        x = self.fc3(x)

        return x

class Shrink(nn.Module):
    def __init__(self, device, num_input):
        super(Shrink, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(12, 7)
        self.bn1 = nn.BatchNorm1d(7)
        self.dp1 = nn.Dropout(p=0.1)

        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 2,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)
        self.concat = AppendTime(device)

    def forward(self, x, t):
        def layer(x, fc, dp, bn):
            return dp(bn(F.relu(fc(x))))
        x = self.rnn(x)[0] 
        x = x.squeeze(1)
        x = self.concat(x,t)
        x = layer(x, self.fc1, self.dp1, self.bn1)
        return x
