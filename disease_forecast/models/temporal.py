import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
    def forward(self, x): 
        T = x.shape[1]
        if T==1:
            return x[:, 0, :] 
        x = nn.LSTM(input_size = x.shape[2],
                hidden_size = x.shape[2], 
                num_layers = 1,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)(x)[0]
        return x[:, -1, :]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
    def forward(self, x): 
        T = x.shape[1]
        if T==1:
            return x[:, 0, :] 
        x = nn.RNN(input_size = x.shape[2],
                hidden_size = x.shape[2], 
                num_layers = 1,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)(x)[0]
        return x[:, -1, :]


