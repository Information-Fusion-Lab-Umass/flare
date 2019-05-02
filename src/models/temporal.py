import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class forecastRNN(nn.Module):
    def __init__(self, device, num_input, num_timesteps):
        super(forecastRNN, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Autoencoder for feature prediction
        self.autoenc = nn.Sequential(
                nn.Linear(num_input, 1000),
                nn.BatchNorm1d(1000),
                nn.Dropout(p=0.5),
                nn.ReLU(),

                nn.Linear(1000, 1000),
                nn.BatchNorm1d(1000),
                nn.Dropout(p=0.5),
                nn.ReLU(),

                nn.Linear(1000, num_input),
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
        return nn.MSELoss()(ypred, y)*1./nelt

    def forward(self, x, t): 
        #  print('in fore RNN ', x.shape)
        x_final = x[:, -1, :]
        x = x[:, :-1, :]
        (bsize, T, nfeat) = x.shape
        #  print('Input feat shape = ', x.shape)
        #  print('Output shape = ', x_final.shape)
        # Forward pass through the RNN
        h = self.rnn(x)[0]
        # Forward pass through the featue prediction model
        h = h.contiguous().view(bsize*T, nfeat)
        y = self.autoenc(h).view(bsize, T, nfeat)
        h = h.view(bsize, T, nfeat)
        #  print(h.is_cuda, y.is_cuda, x.is_cuda)
        #  print('RNN output = {}, feat pred module output = {}'.format\
        #          (h.shape, y.shape))
        #  print(h.min(), h.max(), y.min(), y.max(), x.min(), x.max())
        # Calculate the loss
        lossval = self.loss(y[:,:-1,:], x[:,1:,:]) if T!=1 \
                else torch.tensor(0.).to(self.device)
        #  print('Loss vals = ', lossval)
        #  gap = (t[:,-1] - t[:,-2]).view(-1, 1)
        gap = t.view(-1, 1)
        #  print('Gaps = ', gap.shape, gap.min(), gap.max())
        
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
        #  print(x_pred.is_cuda, gap.is_cuda, x_all_gaps.is_cuda,
        #  print('output shape : ',x_pred.shape)
        lossval += self.loss(x_pred, x_final)
        #  print(lossval.data.numpy(),
        #          x_pred.data.numpy().min(),
        #          x_pred.data.numpy().max(),
        #          x.data.numpy().min(),
        #          x.data.numpy().max(),
        #          x_final.data.numpy().min(),
        #          x_final.data.numpy().max())
        #  lossval = torch.clamp(lossval, min = -3, max = 3)
        #  if lossval.data.numpy() > 10:
        #      print('lossvals = ', self.loss(x_pred, x_final).data.numpy(),
        #              lossval.data.numpy(),
        #              x_final.data.numpy().min(),
        #              x_final.data.numpy().max(),
        #              x_pred.data.numpy().min(),
        #              x_pred.data.numpy().max())
        return x_pred, lossval

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


