import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import pickle

class forecastRNN(nn.Module):
    def __init__(self, device, num_input, num_timesteps):
        super(forecastRNN, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Autoencoder for feature prediction
        self.autoenc = nn.Sequential(
                nn.Linear(num_input, 250),
                nn.BatchNorm1d(250),
                nn.Dropout(p=0.5),
                nn.ReLU(),

                nn.Linear(250, 250),
                nn.BatchNorm1d(250),
                nn.Dropout(p=0.2),
                nn.ReLU(),

                nn.Linear(250, num_input),
                nn.BatchNorm1d(num_input)
                )

        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 1,
                batch_first=True,
                dropout=0.0, 
                bidirectional=False)

    def loss(self, ypred, y):
        nelt = ypred.nelement()
        ypred, y = ypred.to(self.device), y.to(self.device)
        return nn.MSELoss(reduction = 'mean')(ypred, y)*1./nelt

    def forward(self, x, t): 
        x_final = x[:, -1, :]

        x = x[:, :-1, :]
        (bsize, T, nfeat) = x.shape
        # Forward pass through the RNN
        h = self.rnn(x)[0]
        # Forward pass through the feature prediction model
        h = h.contiguous().view(bsize*T, nfeat)
        y = self.autoenc(h).view(bsize, T, nfeat)
        h = h.view(bsize, T, nfeat)
        # Calculate the loss
        lossval = self.loss(y[:,:-1,:], x[:,1:,:]) if T!=1 \
                else torch.tensor(0.).to(self.device)
        gap = t.view(-1, 1)
        
        x_hat = y[:, -1, :]
        h_hat = h[:, -1, :]
        max_gap = int(gap.max()) 
        x_all_gaps = torch.zeros([bsize, max_gap, nfeat]).to(self.device)
        for t_pred in range(max_gap):
            #  print(x_hat.unsqueeze(1).size(), h_hat.unsqueeze(0)
            h_hat = self.rnn(x_hat.unsqueeze(1).contiguous(), \
                    h_hat.unsqueeze(0).contiguous())[0].squeeze()
            if len(h_hat.size()) == 1:
                h_hat = h_hat.unsqueeze(0)
            x_hat = self.autoenc(h_hat)
            x_all_gaps[:, t_pred, :] = x_hat
        gap = (gap - 1)[:,0].long()
        x_pred = x_all_gaps[range(bsize), gap, :]
        lossval += self.loss(x_pred, x_final)
        return x_pred, lossval, y[:,:,:]

class forecastRNN_no_bn(nn.Module):
    def __init__(self, device, num_input, num_timesteps):
        super(forecastRNN_no_bn, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Autoencoder for feature prediction
        self.autoenc = nn.Sequential(
                nn.Linear(num_input, 1000),
                nn.Dropout(p=0.5),
                nn.ReLU(),

                nn.Linear(1000, 1000),
                nn.Dropout(p=0.5),
                nn.ReLU(),

                nn.Linear(1000, num_input)
                )

        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 1,
                batch_first=True,
                dropout=0.0, 
                bidirectional=False)

    def loss(self, ypred, y):
        nelt = ypred.nelement()
        ypred, y = ypred.to(self.device), y.to(self.device)
        return nn.MSELoss(reduction = 'mean')(ypred, y)*1./nelt

    def forward(self, x, t): 
        x_final = x[:, -1, :]

        x = x[:, :-1, :]
        (bsize, T, nfeat) = x.shape
        # Forward pass through the RNN
        h = self.rnn(x)[0]
        # Forward pass through the feature prediction model
        h = h.contiguous().view(bsize*T, nfeat)
        y = self.autoenc(h).view(bsize, T, nfeat)
        h = h.view(bsize, T, nfeat)
        # Calculate the loss
        lossval = self.loss(y[:,:-1,:], x[:,1:,:]) if T!=1 \
                else torch.tensor(0.).to(self.device)
        gap = t.view(-1, 1)
        
        x_hat = y[:, -1, :]
        h_hat = h[:, -1, :]
        max_gap = int(gap.max()) 
        x_all_gaps = torch.zeros([bsize, max_gap, nfeat]).to(self.device)
        for t_pred in range(max_gap):
            #  print(x_hat.unsqueeze(1).size(), h_hat.unsqueeze(0)
            h_hat = self.rnn(x_hat.unsqueeze(1).contiguous(), \
                    h_hat.unsqueeze(0).contiguous())[0].squeeze()
            if len(h_hat.size()) == 1:
                h_hat = h_hat.unsqueeze(0)
            x_hat = self.autoenc(h_hat)
            x_all_gaps[:, t_pred, :] = x_hat
        gap = (gap - 1)[:,0].long()
        x_pred = x_all_gaps[range(bsize), gap, :]
        lossval += self.loss(x_pred, x_final)
        return x_pred, lossval, y[:,:,:]

class forecastRNN_covtest(nn.Module):
    def __init__(self, device, num_input, num_timesteps):
        super(forecastRNN_covtest, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Autoencoder for feature prediction
        self.autoenc = nn.Sequential(
                nn.Linear(num_input, 250),
                nn.BatchNorm1d(250),
                nn.Dropout(p=0.3),
                nn.ReLU(),

                nn.Linear(250, 250),
                nn.BatchNorm1d(250),
                nn.Dropout(p=0.5),
                nn.ReLU(),

                nn.Linear(250, num_input),
                nn.BatchNorm1d(num_input)
                )

        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 1,
                batch_first=True,
                dropout=0.0, 
                bidirectional=False)

    def loss(self, ypred, y):
        nelt = ypred.nelement()
        ypred, y = ypred.to(self.device), y.to(self.device)
        return nn.MSELoss(reduction = 'mean')(ypred, y)*1./nelt

    def forward(self, x, t): 
        x_final = x[:, -1, :]
        
        x = x[:, :-1, :]
        (bsize, T, nfeat) = x.shape
        # Forward pass through the RNN
        h = self.rnn(x)[0]
        #if(T == 1):
        #    return h

        # Forward pass through the feature prediction model
        h = h.contiguous().view(bsize*T, nfeat)
        y = self.autoenc(h).view(bsize, T, nfeat)
        h = h.view(bsize, T, nfeat)
        # Calculate the loss
        lossval = self.loss(y[:,:-1,:], x[:,1:,:]) if T!=1 \
                else torch.tensor(0.).to(self.device)
        gap = t.view(-1, 1)
        
        x_hat = y[:, -1, :]
        h_hat = h[:, -1, :]
        max_gap = int(gap.max()) 
        x_all_gaps = torch.zeros([bsize, max_gap, nfeat]).to(self.device)
        for t_pred in range(max_gap):
            #  print(x_hat.unsqueeze(1).size(), h_hat.unsqueeze(0)
            h_hat = self.rnn(x_hat.unsqueeze(1).contiguous(), \
                    h_hat.unsqueeze(0).contiguous())[0].squeeze()
            if len(h_hat.size()) == 1:
                h_hat = h_hat.unsqueeze(0)
            x_hat = self.autoenc(h_hat)
            x_all_gaps[:, t_pred, :] = x_hat
        gap = (gap - 1)[:,0].long()
        x_pred = x_all_gaps[range(bsize), gap, :]
        lossval += self.loss(x_pred, x_final)
        return x_pred, lossval, y[:,:,:]

class RNN(nn.Module):
    def __init__(self, device, num_input, num_timesteps):
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

