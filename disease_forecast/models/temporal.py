import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, num_input, num_timesteps):
        super(RNN, self).__init__()
        self.T = num_timesteps
        self.lstm = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 1,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)
        
    def forward(self, x): 
        if self.T==1:
            return x[:, 0, :] 
        x = self.lstm(x)[0]
        return x[:, -1, :]

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


