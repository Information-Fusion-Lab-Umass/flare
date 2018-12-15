import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class forecastRNN(nn.Module):
    def __init__(self, num_input, num_timesteps):
        super(forecastRNN, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.num_timesteps = num_timesteps
        
        # Autoencoder for feature prediction
        self.autoenc = nn.Sequential(
                nn.Linear(num_input, 1000),
                nn.BatchNorm1d(1000),
                nn.Dropout(p=0.5),

                nn.Linear(1000, 1000),
                nn.BatchNorm1d(1000),
                nn.Dropout(p=0.5),

                nn.Linear(1000, num_input),
                nn.BatchNorm1d(num_input)
                )

        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 2,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)
    
    def loss(self, ypred, y):
        return nn.MSELoss()(ypred, y)

    def forward(self, x, t): 
        (bsize, T, nfeat) = x.shape
        if self.T==1:
            return x[:, 0, :] 
        # Forward pass through the RNN
        h = self.rnn(x)[0]
        # Forward pass through the featue prediction model
        h = h.contiguous().view(bsize*T, nfeat)
        y = self.autoenc(h).view(bsize, T, nfeat)
        print(h.shape, y.shape)
        # Calculate the loss
        lossval = self.loss(y[:, :-1, :], x[:, 1:, :])
        x_hat = y[:, -1, :].unsqueeze(1)
        gap = (t[:,-1] - t[:,-2]).view(-1, 1)
        #  x_hat = torch.zeros((
        for t_pred in range(6-T):
            h_hat = self.rnn(x_hat)[0]
            h_hat = h_hat.contiguous().view(bsize, nfeat).squeeze(1)
            x_hat = self.autoenc(h_hat)        
        print(gap.shape, x_hat.shape, gap.min(), gap.max())
        return x[:, -1, :]

class RNN(nn.Module):
    def __init__(self, num_input, num_timesteps):
        super(RNN, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 2,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)
        
    def forward(self, x): 
        if self.T==1:
            return x[:, 0, :] 
        x = self.rnn(x)[0]
        return x[:, -1, :]

class LSTM(nn.Module):
    def __init__(self, num_input, num_timesteps):
        super(LSTM, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.lstm = nn.LSTM(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 3,
                batch_first=True,
                dropout=0.1, 
                bidirectional=False)
        
    def forward(self, x): 
        if self.T==1:
            return x[:, 0, :] 
        x = self.lstm(x)[0]
        return x[:, -1, :]


