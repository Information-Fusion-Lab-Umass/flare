import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from time import time
import torch
import torch.nn as nn
import yaml
import ipdb
import pickle
from src import models, utils, datagen, evaluate, unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#  torch.backends.cudnn.enabled = False

class Model(nn.Module):
    def __init__(self, class_wt, module_image, module_temporal, \
            module_forecast, module_task, fusion):
        super(Model, self).__init__()
        # Model names file
        with open('../src/models/models.yaml') as f:
            model_dict = yaml.load(f)
        self.fusion = fusion

        # Load model: image architecture
        self.model_image_name = module_image.pop('name')
        self.model_image = eval(model_dict[self.model_image_name])(**module_image)
        
        if self.fusion == 'concat_feature':
            # Load model: longitudinal architecture
            self.model_long = eval(model_dict['long'])()

            # Load model: covariate architecture
            self.model_cov = eval(model_dict['cov'])()

        # Load model: temporal architecture
        self.model_temporal_name = module_temporal.pop('name')
        self.model_temporal = eval(model_dict[self.model_temporal_name])(**module_temporal)
        
        # Load model: forecast architecture
        model_forecast_name = module_forecast.pop('name')
        self.model_forecast = eval(model_dict[model_forecast_name])(**module_forecast)

        # Load model: Task specific architecture
        model_task_name = module_task.pop('name')
        self.model_task = eval(model_dict[model_task_name])(**module_task)

        # Class weights for loss
        self.class_wt = torch.tensor(class_wt).float()

    def loss(self, y_pred, y):
        return nn.CrossEntropyLoss(weight=self.class_wt)(y_pred, y.long())
 
    def forward(self, data_batch, on_gpu=False):
        # Extract data components
        x_img_data = data_batch['img_features']
        x_cov_data = data_batch['covariates']
        x_long_data = data_batch['test_scores']
        x_time_data = data_batch['tau']
        (B, T, _) = x_img_data.shape

        # STEP 3: MODULE 1: FEATURE EXTRACTION -----------------------------
        # Get image features :  x_img_feat = (B, T-1, Fi) 
        x_img_data = x_img_data.view((B*(T), 1) + x_img_data.shape[2:])
        if len(x_img_data.shape) == 5:
            x_img_data = x_img_data.permute(0,1,4,2,3)
        if self.fusion == 'concat_feature':
            x_img_feat = self.model_image(x_img_data)
            x_img_feat = x_img_feat.view(B, T, -1)

            # Get longitudinal features : x_long_feat: (B, T-1, Fl)
            x_long_data = x_long_data.view(B*(T), -1)
            x_long_feat = self.model_long(x_long_data)
            x_long_feat = x_long_feat.view(B, T, -1)
      
            # Get Covariate features : x_cov_feat: (B, T-1, Fc)
            x_cov_data = x_cov_data.view(B*(T), -1)
            x_cov_feat = self.model_cov(x_cov_data)
            x_cov_feat = x_cov_feat.view(B, T, -1)
 
        elif self.fusion == 'concat_input':
            x_img_data = x_img_data.view(B, T ,-1) 
            x_img_data = torch.cat((x_img_data, x_long_data, x_cov_data), -1)
            x_img_data = x_img_data.view(B*T,-1)
            x_img_feat = self.model_image(x_img_data)
            x_img_feat = x_img_feat.view(B, T, -1)


        # STEP 4: MULTI MODAL FEATURE FUSION -------------------------------
        # Fuse the features
        # x_feat: (B, T-1, F_i+F_l+F_c) = (B, T-1, F)
        if self.fusion == 'concat_feature':
            x_feat = torch.cat((x_img_feat, x_long_feat, x_cov_feat), -1)
        elif self.fusion == 'concat_input':
            x_feat = x_img_feat

        # STEP 5: MODULE 2: TEMPORAL FUSION --------------------------------
        # X_temp: (B, F_t)
        if self.model_temporal_name == 'forecastRNN':
            x_forecast, lossval = self.model_temporal(x_feat, x_time_data)
            #  print('auxloss = ', lossval.data.numpy())
        else:
            x_temp = self.model_temporal(x_feat[:, :-1, :])
      
            # STEP 6: MODULE 3: FORECASTING ------------------------------------
            # x_forecast: (B, F_f)
            x_forecast = self.model_forecast(x_temp, x_time_data)
            lossval = torch.tensor(0.)        
 
        # STEP 7: MODULE 4: TASK SPECIFIC LAYERS ---------------------------
        # DX Classification Module
        ypred = self.model_task(x_forecast)

        return ypred, lossval

class Engine:
    def __init__(self, model_config):

        load_model = model_config.pop('load_model')
        self.num_classes = model_config['module_task']['num_classes']
        self.model = Model(**model_config)

        # Load the model
        if load_model != '':
            self.model.load_state_dict(torch.load(load_model))

        #  self.model.cuda()
        self.model_params = {
                'image': list(self.model.model_image.parameters()),
                'temporal': list(self.model.model_temporal.parameters()),
                'task': list(self.model.model_task.parameters())
                }
        if self.model.fusion == 'concat_feature':
            self.model_params['long'] = list(self.model.model_long.parameters())
            self.model_params['cov'] = list(self.model.model_cov.parameters())

        # Initialize the optimizer
        self.optm = torch.optim.Adam(sum(list(self.model_params.values()), []))

    def train(self, datagen_train, datagen_val, \
            exp_dir, num_epochs, log_period=100, \
            ckpt_period=100, validation_period = 100, save_model=False):
        
        loss_vals = evaluate.LossVals(
                num_epochs,
                validation_period,
                len(datagen_train)
                )
        #  loss_vals = {}
        #  loss_vals['train_loss_T'] = np.zeros((num_epochs, len(datagen_train)))
        #  loss_vals['train_loss'] = np.zeros((num_epochs))
        #  loss_vals['val_loss_T'] = np.zeros((num_epochs/validation_period, \
        #          len(datagen_train)))
        #  loss_vals['val_loss'] = np.zeros((num_epochs/validation_period))
        #  idx_val = 0

        for epoch in range(num_epochs):
            # TRAIN THE MODEL ---------------------------
            self.model.train()

            # Clone params for module-wise unit test
            params = {}
            for key in self.model_params:
                p = self.model_params[key]
                params[key] = [p[i].clone() for i in range(len(p))]

            # Iterate over datagens for T = [2, 3, 4, 5, 6]

            lossval = 0.0; count = 0
            for idx, datagen in enumerate(datagen_train):
                t = time()
                clfLoss_T = 0.0 ; auxLoss_T = 0.0
                for step, (x, y) in enumerate(datagen):
                    self.optm.zero_grad() 
                    # Feed Forward
                    y_pred, auxloss = self.model(x)
                    clfloss = self.model.loss(y_pred, y)
                    obj = clfloss + auxloss
                    # Train the model
                    obj.backward()
                    self.optm.step()
                    clfLoss_T += float(clfloss)
                    auxLoss_T += float(auxloss)

                    if step == 5:
                        break
                print('Epoch = {}, datagen = {}, steps = {}, time = {}'.\
                        format(epoch, idx, step, time() - t))

                # Store the Loss
                loss_vals.update_T('train', [clfLoss_T, auxLoss_T], \
                        epoch, idx, step + 1)
            loss_vals.update('train', epoch)

            # Unittest
            unittest.change_in_params(params, self.model_params)
 
            # VALIDATION ------------------------------------
            if epoch % validation_period == 0:
                self.model.eval()
                for idx, datagen in enumerate(datagen_val):
                    clfLoss_T = 0.0 ; auxLoss_T = 0.0
                    for step, (x, y) in enumerate(datagen):
                        # Feed Forward
                        y_pred, auxloss = self.model(x)
                        clfloss = self.model.loss(y_pred, y)
                        obj = clfloss + auxloss
                        #  print('val = ',clfloss.data.numpy(), auxloss.data.numpy(), \
                        #          obj.data.numpy())
                        # Store the validation loss
                        clfLoss_T += float(clfloss)
                        auxLoss_T += float(auxloss)

                        if step == 5:
                            break

                    # Store the Loss
                    loss_vals.update_T('val', [clfLoss_T, auxLoss_T], \
                        int(epoch/validation_period), idx, step + 1)
                loss_vals.update('val', int(epoch/validation_period))

            # LOGGING --------------------------------------
            if epoch % log_period == 0:
                print('Epoch : {}, Train Loss = {}'. \
                        format(epoch + 1, loss_vals.train_loss['totalLoss'][epoch]))

            # SAVING THE MODEL -----------------------------
            if epoch % ckpt_period == 0:
                print('Checkpoint : Saving model at Epoch : {}'.\
                        format(epoch+1))
                torch.save(self.model.state_dict(), exp_dir + \
                        '/checkpoints/model_ep' + str(epoch+1) + '.pth')




        # SAVING THE LOGS -----------------------------------
        with open(os.path.join(exp_dir, 'logs/loss.pickle'), 'wb') as f:
            pickle.dump(loss_vals, f)

        loss_vals.plot_graphs(os.path.join(exp_dir, 'logs'))


    def test(self, data, exp_dir, data_name, batch_size, feat_flag):

        self.model.eval()
        cnf_matrix = np.empty((4, 5), dtype=object)
        for n_t in range(1, 5):
            data_t = datagen.get_timeBatch(data, n_t, feat_flag)
            time_t = datagen.get_time_batch(data_t, as_tensor=True)
            time_t = (time_t[:,-1] - time_t[:,-2]).data.numpy()
            (N, T) = data_t.shape
            num_batches = int(N/batch_size)
            for i in range(num_batches):
                data_t_batch = data_t[i*batch_size:(i+1)*batch_size]
                y_pred_i, _ = self.model(data_t_batch)
                y_dx_i = datagen.get_labels(data_t_batch, \
                        task='dx', as_tensor=True)
                if i == 0:
                    y_pred, y_dx = y_pred_i, y_dx_i
                else:
                    y_pred = torch.cat((y_pred, y_pred_i), 0) 
                    y_dx = torch.cat((y_dx, y_dx_i), 0) 
            data_t_batch = data_t[num_batches*batch_size:]                        
            if data_t_batch.shape[0]>1:
                y_pred = torch.cat((y_pred, self.model(data_t_batch)[0]), 0)
                y_dx = torch.cat((y_dx, datagen.get_labels(data_t_batch, \
                        task='dx', as_tensor=True)), 0)            
            for t in range(6-n_t):
                idx = np.where(time_t[:len(y_dx)]==t+1) 
                cnf_matrix[n_t-1, t] = evaluate.cmatCell(
                        evaluate.confmatrix_dx(y_pred[idx], y_dx[idx], self.num_classes))

            evaluate.get_output(cnf_matrix, exp_dir, data_name, 'dx', self.num_classes)

        
