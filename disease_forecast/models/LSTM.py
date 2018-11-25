import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
    def forward(self, x, output='tensor'): 
        x = torch.from_numpy(x).float()
        x = nn.LSTM(input_size = x.shape[2],
                hidden_size = x.shape[2], 
                num_layers = 1,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)(x)
        x = x[:, -1, :]
        if output=='numpy':
            x = x.data.numpy()
        return x


