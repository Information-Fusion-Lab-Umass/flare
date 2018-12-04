import numpy as np
import os
import pandas as pd
from scipy.misc import imsave
from tqdm import tqdm
import torch
import torch.nn as nn
import ipdb
from disease_forecast import models, utils, datagen, evaluate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, module_image, module_temporal, \
            module_forecast, module_task, fusion):
        super(Model, self).__init__()
        # Model Architectures: Image
        model_dict = {
                'tadpole1': models.Tadpole1,
                'tadpole2': models.Tadpole2,
                'long': models.Longitudinal,
                'cov': models.Covariate,
                'lstm': models.LSTM,
                'rnn': models.RNN,
                'append_time' : models.AppendTime,
                'multiply_time' : models.MultiplyTime,
                'dx': models.ANN_DX 
                }
        
        # Load model: image architecture
        model_image_name = module_image.pop('name')
        self.model_image = model_dict[model_image_name](**module_image)
        
        # Load model: longitudinal architecture
        self.model_long = model_dict['long']()

        # Load model: covariate architecture
        self.model_cov = model_dict['cov']()

        # Load model: temporal architecture
        model_temporal_name = module_temporal.pop('name')
        self.model_temporal = model_dict[model_temporal_name](**module_temporal)
        
        # Load model: forecast architecture
        model_forecast_name = module_forecast.pop('name')
        self.model_forecast = model_dict[model_forecast_name](**module_forecast)

        # Load model: Task specific architecture
        model_task_name = module_task.pop('name')
        self.model_task = model_dict[model_task_name](**module_task)

        self.fusion = fusion

    def loss(self, y_pred, y):
        return self.model_task.loss(y_pred, y)
        
    def forward(self, data_batch):
        (B, T) = data_batch.shape
        T = 2 if T == 1 else T
        # STEP 2: EXTRACT INPUT VALUES -------------------------------------
        # Get time data : x_time_data = (B, T)
        x_time_data = datagen.get_time_batch(data_batch, as_tensor=True)
        # Get image data : x_img_data: (B, T-1, Di)
        x_img_data = datagen.get_img_batch(data_batch, as_tensor=True) 
        # Get longitudinal data : x_long_data: (B, T-1, Dl)  
        x_long_data = datagen.get_long_batch(data_batch, as_tensor=True)
        # Get covariate data : x_cov_data: (B, T-1, Dc)
        x_cov_data = datagen.get_cov_batch(data_batch, as_tensor=True)
        #  print('Input data dims: Time={}, Image={}, Long={}, Cov={}'.\
        #             format(x_time_data.shape, x_img_data.shape, \
        #             x_long_data.shape, x_cov_data.shape))
        
        # STEP 3: MODULE 1: FEATURE EXTRACTION -----------------------------
        # Get image features :  x_img_feat = (B, T-1, Fi) 
        x_img_data = x_img_data.view(B*(T-1), 1, -1)
        x_img_feat = self.model_image(x_img_data)
        x_img_feat = x_img_feat.view(B, T-1, -1)
        #  print(x_img_feat.min(), x_img_feat.max())
        #  print('Image features dim = ', x_img_feat.shape)
 
        # Get longitudinal features : x_long_feat: (B, T-1, Fl)
        x_long_data = x_long_data.view(B*(T-1), -1)
        x_long_feat = self.model_long(x_long_data)
        x_long_feat = x_long_feat.view(B, T-1, -1)
        #  print(x_long_feat.min(), x_long_feat.max())
        #  print('Longitudinal features dim = ', x_long_feat.shape)
 
        # Get Covariate features : x_cov_feat: (B, T-1, Fc)
        x_cov_data = x_cov_data.view(B*(T-1), -1)
        x_cov_feat = self.model_cov(x_cov_data)
        x_cov_feat = x_cov_feat.view(B, T-1, -1)
        #  print(x_cov_feat.min(), x_cov_feat.max())
        #  print('Covariate features dim = ', x_cov_feat.shape)
 
        # STEP 4: MULTI MODAL FEATURE FUSION -------------------------------
        # Fuse the features
        # x_feat: (B, T-1, F_i+F_l+F_c) = (B, T-1, F)
        if self.fusion=='latefuse':
            x_feat = torch.cat((x_img_feat, x_long_feat, x_cov_feat), -1)
        #  print('Feature fusion dims = ', x_feat.shape)
        #  print(x_feat.min(), x_feat.max())
 
        # STEP 5: MODULE 2: TEMPORAL FUSION --------------------------------
        # X_temp: (B, F_t)
        x_temp = self.model_temporal(x_feat)
        #  print('Temporal dims = ', x_temp.shape)
 
        # STEP 6: MODULE 3: FORECASTING ------------------------------------
        # x_forecast: (B, F_f)
        x_forecast = self.model_forecast(x_temp, x_time_data)
        #  print('Forecast dims = ', x_forecast.shape)
 
        # STEP 7: MODULE 4: TASK SPECIFIC LAYERS ---------------------------
        # DX Classification Module
        ypred = self.model_task(x_forecast)
        return ypred

class Engine:
    def __init__(self, model_config):

        load_model = model_config.pop('load_model')
        self.model = Model(**model_config)

        # Load the model
        if load_model != '':
            self.model.load_state_dict(torch.load(load_model))

        # Initialize the optimizer
        self.optm = torch.optim.Adam(
                #  list(self.model.model_image.parameters()) + \
                list(self.model.model_long.parameters()) + \
                list(self.model.model_cov.parameters()) + \
                list(self.model.model_temporal.parameters()) + \
                #  list(self.model.model_forecast.parameters()) + \
                list(self.model.model_task.parameters())
                )

    def train(self, datagen_train, datagen_val, exp_dir, \
            num_epochs, save_model=False):
        
        print('Training ...')
        loss_vals = np.zeros((num_epochs, 3))

        # For each epoch,
        for epoch in range(num_epochs):
            self.optm.zero_grad() 
            self.model.train()
            
            # Get Train data loss
            x_train_batch = next(datagen_train)
            y_dx = datagen.get_labels(x_train_batch, task='dx', as_tensor=True)
            y_pred = self.model(x_train_batch)
            obj = self.model.loss(y_pred, y_dx)

            # Train the model
            obj.backward() 
            self.optm.step()

            # Get validation loss
            self.model.eval()
            x_val_batch = next(datagen_val)
            y_val_dx = datagen.get_labels(x_val_batch, task='dx', as_tensor=True)
            y_val_pred = self.model.forward(x_val_batch)
            loss_val = self.model.loss(y_val_pred, y_val_dx)

            # print loss values
            loss_vals[epoch, :] = [
                    obj.data.numpy(),
                    loss_val.data.numpy(),
                    x_train_batch.shape[1]
                    ]
            if epoch%100 == 0:
                print('Loss at epoch {} = {}, T = {}'. format(epoch+1, \
                        obj.data.numpy(), x_train_batch.shape[1]))

        # Save loss values as csv and plot image
        df = pd.DataFrame(loss_vals)
        df.columns = ['loss_train', 'loss_val', 'time_forecast']
        df.to_csv(exp_dir+'/logs/loss.csv')

        plt.figure()
        plt.plot(np.arange(num_epochs), loss_vals[:,0], c='b', label='Train Loss')
        plt.plot(np.arange(num_epochs), loss_vals[:,1], c='r', label='Validation Loss')
        plt.title('Train and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(exp_dir+'/logs/loss.png', dpi=300)
        plt.close()

        # Save the model
        if save_model:
            torch.save(self.model.state_dict(), exp_dir+'/checkpoints/model.pth')      

    def test(self, data, exp_dir, data_type, task, data_split, batch_size, feat_flag):
        if task == 'forecast':
            cnf_matrix = np.empty((4, 5), dtype=object)
            self.model.eval()
            for n_t in range(1, 5):
                data_t = datagen.get_timeBatch(data, n_t, feat_flag)
                time_t = datagen.get_time_batch(data_t, as_tensor=True)
                time_t = (time_t[:,-1] - time_t[:,-2]).data.numpy()
                (N, T) = data_t.shape
                num_batches = int(N/batch_size)
                for i in range(num_batches):
                    data_t_batch = data_t[i*batch_size:(i+1)*batch_size]
                    y_pred_i = self.model(data_t_batch)
                    y_dx_i = datagen.get_labels(data_t_batch, \
                            task='dx', as_tensor=True)
                    if i == 0:
                        y_pred, y_dx = y_pred_i, y_dx_i
                    else:
                        y_pred = torch.cat((y_pred, y_pred_i), 0) 
                        y_dx = torch.cat((y_dx, y_dx_i), 0) 
                data_t_batch = data_t[num_batches*batch_size:]                        
                if data_t_batch.shape[0]>1:
                    y_pred = torch.cat((y_pred, self.model(data_t_batch)), 0)
                    y_dx = torch.cat((y_dx, datagen.get_labels(data_t_batch, \
                            task='dx', as_tensor=True)), 0)            
                for t in range(6-n_t):
                    idx = np.where(time_t[:len(y_dx)]==t+1) 
                    cnf_matrix[n_t-1, t] = evaluate.cmatCell(
                            evaluate.confmatrix_dx(y_pred[idx], y_dx[idx]))
            evaluate.get_output(cnf_matrix, exp_dir, data_type, 'dx')
        elif task=='classify':
            self.model.eval()
            data_t = datagen.get_timeBatch(data, 0, feat_flag)
            N = data_t.shape[0]
            num_batches = int(N/batch_size)
            for i in range(num_batches):
                data_t_batch = data_t[i*batch_size:(i+1)*batch_size]
                y_pred_i = self.model(data_t_batch)
                y_dx_i = datagen.get_labels(data_t_batch, \
                        task='dx', as_tensor=True)
                if i == 0:
                    y_pred, y_dx = y_pred_i, y_dx_i
                else:
                    y_pred = torch.cat((y_pred, y_pred_i), 0) 
                    y_dx = torch.cat((y_dx, y_dx_i), 0) 
            data_t_batch = data_t[num_batches*batch_size:]                        
            if data_t_batch.shape[0]>1:
                y_pred = torch.cat((y_pred, self.model(data_t_batch)), 0)
                y_dx = torch.cat((y_dx, datagen.get_labels(data_t_batch, \
                        task='dx', as_tensor=True)), 0)
            cnf_matrix = evaluate.cmatCell(
                    evaluate.confmatrix_dx(y_pred, y_dx))
            evaluate.get_output(cnf_matrix, exp_dir, data_type, 'dx')

