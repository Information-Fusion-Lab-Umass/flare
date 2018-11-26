import torch
import torch.nn as nn
import torch.nn.functional as F

class AppendTime(nn.Module):
    def __init__(self):
        super(AppendTime, self).__init__()
        
    def forward(self, x, output='tensor'): 
        x = torch.from_numpy(x).float()
        print(x.shape)
        x = nn.LSTM(input_size = x.shape[2],
                hidden_size = x.shape[2], 
                num_layers = 1,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)(x)[0]
        print(x.shape)
        x = x[:, -1, :]
        if output=='numpy':
            x = x.data.numpy()
        return x


