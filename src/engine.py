import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from time import time
import torch
import torch.nn as nn
import yaml
#import ipdb
import pickle
from src import models, utils, evaluate, unittest_
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import random
#  torch.backends.cudnn.enabled = False

class Model(nn.Module):
    def __init__(self, device, class_wt, image, temporal, \
            forecast, task, fusion, aux_loss="MSE", aux_loss_scale=1.0):
        super(Model, self).__init__()
        # Model names file
        with open('../src/models/models.yaml') as f:
            model_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.fusion = fusion
        self.device = device
        self.aux_loss = aux_loss #model_config.pop('aux_loss','MSE')
        self.aux_loss_scale = aux_loss_scale

        # Load model: image architecture
        self.model_image_name = image['name']
        self.model_image = eval(model_dict[self.model_image_name])(**image['params'])
        self.model_image = self.model_image.to(device)
        
        if self.fusion == 'concat_feature':
            # Load model: longitudinal architecture
            self.model_long = eval(model_dict['long'])()
            self.model_long = self.model_long.to(device)

            # Load model: covariate architecture
            self.model_cov = eval(model_dict['cov'])()
            self.model_cov = self.model_cov.to(device)

        # Load model: temporal architecture
        self.model_temporal_name = temporal['name']
        self.model_temporal = eval(model_dict[self.model_temporal_name])\
                (device, **temporal['params'])
        self.model_temporal = self.model_temporal.to(device)
        
        # Load model: forecast architecture
        model_forecast_name = forecast['name']
        self.model_forecast = eval(model_dict[model_forecast_name])\
                (device) #**forecast)
        self.model_forecast = self.model_forecast.to(device)

        # Load model: Task specific architecture
        model_task_name = task['name']
        self.model_task = eval(model_dict[model_task_name])(**task['params'])
        self.model_task = self.model_task.to(device)

        self.class_wt = torch.tensor(class_wt).float().to(device)
        
    def loss(self, y_pred, y):
        y = y.long().to(self.device)
        return nn.CrossEntropyLoss(weight = self.class_wt)(y_pred, y)
 
    def forward(self, img_features, covariates, test_scores, tau, labels, on_gpu=False):
        # Extract data components
        x_img_data = img_features
        x_cov_data = covariates
        x_long_data = test_scores
        x_time_data = tau
        x_labels = labels
        (B, T, _) = x_img_data.shape
       
        # STEP 3: MODULE 1: FEATURE EXTRACTION -----------------------------
        # Get image features :  x_img_feat = (B, T-1, Fi) 
        x_img_data = x_img_data.view((B*(T), 1) + x_img_data.shape[2:])
        if len(x_img_data.shape) == 5:
            x_img_data = x_img_data.permute(0,1,4,2,3)
        if self.fusion == 'concat_feature':
                
           # print('Img Input Max: ', np.max(x_img_data.cpu().detach().numpy()))
           # print('Img Input Min: ', np.min(np.abs(x_img_data.cpu().detach().numpy())))
            x_img_feat = self.model_image(x_img_data)
            x_img_feat = x_img_feat.view(B, T, -1)

           # print('Img Feat Max: ', np.max(x_img_feat.cpu().detach().numpy()))
           # print('Img Feat Min: ', np.min(x_img_feat.cpu().detach().numpy()))
            
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
            # print('Feat Max: ', np.max(x_feat.cpu().detach().numpy()))
            # print('Feat Min: ', np.min(x_feat.cpu().detach().numpy()))
        elif self.fusion == 'concat_input':
            x_feat = x_img_feat

        # STEP 5: MODULE 2: TEMPORAL FUSION --------------------------------
        # X_temp: (B, F_t)
#        print('Shape of x_feat: ',x_feat.shape)
        if self.model_temporal_name[:11] == 'forecastRNN':
            x_forecast, lossval, x_cache = self.model_temporal(x_feat, x_time_data)

        else:
            x_temp = self.model_temporal(x_feat[:, :-1, :])
      
            # STEP 6: MODULE 3: FORECASTING ------------------------------------
            # x_forecast: (B, F_f)
            x_forecast = self.model_forecast(x_temp, x_time_data)
            lossval = torch.tensor(0.).to(self.device)        
 
        # STEP 7: MODULE 4: TASK SPECIFIC LAYERS ---------------------------
        # DX Classification Module
        ypred = self.model_task(x_forecast)
        
        # STEP 8: Compute Auxiliary Loss
        if self.model_temporal_name == 'forecastRNN' and self.aux_loss == "cross_entropy":
            lossval = torch.tensor(0.).to(self.device)
            if T != 2:
                for i in range(T-1):
                    ypred_aux = self.model_task(x_cache[:,i,:])
                    lossval += self.loss(ypred_aux, x_labels[:,i+1,0])
        return ypred, self.aux_loss_scale * lossval

class Engine:
    def __init__(self, class_wt, model_config):
        load_model = model_config.pop('load_model')
        self.early_stopping = model_config.pop('early_stopping')
        self.num_classes = model_config['module_task']['num_classes']
        self.lr = model_config.pop('learning_rate')
        self.weight_decay = model_config.pop('weight_decay')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        init = model_config.pop('init')
        rnn_init = init['rnn_init']
        linear_init = init['linear_init']
        self.aux_loss_scale = model_config['aux_loss_scale']
        self.model = Model(self.device, class_wt, **model_config).to(self.device)

        # initialize weights
        if rnn_init:
            self.model.apply(utils.init_rnn)
        if linear_init:
            self.model.apply(utils.init_kaiming_uniform)

        # Load the model
        if load_model != '':
            self.model.load_state_dict(
                    torch.load(load_model, map_location = self.device)
                    )

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
        self.optm = torch.optim.Adam(sum(list(self.model_params.values()), []), lr = self.lr, weight_decay=self.weight_decay)

    def train(self, datagen_train, datagen_val, \
            exp_dir, dataload_method, data_train_size, batch_size, num_epochs, \
            log_period=100, ckpt_period=100, \
            validation_period = 100, save_model=False):
        
        loss_vals = evaluate.LossVals(
                num_epochs,
                validation_period,
                len(datagen_train)
                )
        # Set min for early stopping condition
        min_loss = np.inf

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
            
            #Check if running T subset exp 
            T_subset = (len(datagen_train) == 1)
            
            for idx, datagen in enumerate(datagen_train):
                t = time()
                clfLoss_T = 0.0; auxLoss_T = 0.0
                for step, (x,y) in enumerate(datagen):
                    self.optm.zero_grad()

                    x = {k : v.to(self.device) for k, v in x.items()}
                    y = y.to(self.device)
                    y_pred, auxloss = self.model(x)
                    clfloss = self.model.loss(y_pred, y)

                    if self.model.model_temporal_name == 'forecastRNN' and not T_subset:
                        if idx == 0:
                            obj = clfloss + auxloss*(1+self.aux_loss_scale)
                        else:
                            obj = clfloss + auxloss
                    else:
                        obj = clfloss + auxloss
                    
                    # Train the model
                    obj.backward()
                    self.optm.step()

                    clfLoss_T += float(clfloss)
                    auxLoss_T += float(auxloss)

                    sys.stdout.flush()
                    
                if epoch == 0:
                    print('Epoch = {}, datagen = {}, steps = {}, time = {}'.\
                           format(epoch, idx, step, time() - t))
                
                loss_vals.update_T('train', [clfLoss_T, auxLoss_T], epoch, idx, step + 1)
                print('auxLoss_T: ', auxLoss_T)
                    
            loss_vals.update('train', epoch)

            # Unittest
            unittest.change_in_params(params, self.model_params)
 
            # VALIDATION ------------------------------------
            print('Validation')
            if(epoch % validation_period == 0 or epoch == num_epochs - 1):
                self.model.eval()
                
                for idx, datagen in enumerate(datagen_val):   
                    t = time()
                    clfLoss_T = 0.0 ; auxLoss_T = 0.0
                    for step, (x, y) in enumerate(datagen):
                        # Feed Forward
                        x = {k : v.to(self.device) for k, v in x.items()}
                        y = y.to(self.device)
                        y_pred, auxloss = self.model(x)
                        clfloss = self.model.loss(y_pred, y)
                        obj = clfloss + auxloss

                        # Store the validation loss
                        clfLoss_T += float(clfloss)
                        auxLoss_T += float(auxloss)
                        sys.stdout.flush()

                    if epoch == 0:
                        print('Epoch = {}, datagen = {}, steps = {}, time = {}'.\
                                format(epoch, idx, step, time() - t))

                    # Store the Loss
                    loss_vals.update_T('val', [clfLoss_T, auxLoss_T], \
                        int(epoch/validation_period), idx, step + 1)  

                loss_vals.update('val', int(epoch/validation_period))

                if loss_vals.val_loss['totalLoss'][epoch] < min_loss:
                    # SAVING THE MODEL -----------------------------
                    min_loss = loss_vals.val_loss['totalLoss'][epoch]
                    if(save_model):
                        if(epoch % ckpt_period == 0 or epoch == num_epochs - 1):
                            print('Checkpoint : Saving model at Epoch : {}'.\
                                    format(epoch+1))
                            torch.save(self.model.state_dict(), exp_dir + \
                                    '/checkpoints/model_ep_min' + '.pth')

            # LOGGING --------------------------------------
            if epoch % log_period == 0:
                print('Epoch : {}, Train Loss = {}'. \
                        format(epoch + 1, loss_vals.train_loss['totalLoss'][epoch]))

            sys.stdout.flush()

        # SAVING THE LOGS -----------------------------------
        with open(os.path.join(exp_dir, 'logs/loss.pickle'), 'wb') as f:
            pickle.dump(loss_vals, f)

        loss_vals.plot_graphs(os.path.join(exp_dir, 'logs'), \
                num_graphs = len(datagen_train))

    def test(self, datagen_test, exp_dir, filename):
        if self.early_stopping: 
            load_model = exp_dir+'/checkpoints/model_ep_min.pth'
            self.model.load_state_dict(torch.load(load_model, map_location = self.device))
        self.model.eval()
        numT = len(datagen_test)
        cnf_matrix = evaluate.ConfMatrix(numT, self.num_classes)
            
        for idx, datagen in enumerate(datagen_test):
            for step, (x_batch, y_batch) in enumerate(datagen):
                x_batch = {k : v.to(self.device) for k, v in x_batch.items()}
                y_batch = y_batch.to(self.device)
                y_pred_batch, auxloss = self.model(x_batch)
                clfloss = self.model.loss(y_pred_batch,y_batch)
                y_pred_batch = nn.Softmax(dim = 1)(y_pred_batch)

                if step == 0:
                    y_pred, y, tau = y_pred_batch, y_batch, x_batch['tau']
                else:
                    y_pred = torch.cat((y_pred, y_pred_batch), 0)
                    y = torch.cat((y, y_batch), 0)
                    tau = torch.cat((tau, x_batch['tau']), 0)

            tau = tau.cpu().data.numpy()
            for t in range(numT - idx):
                loc = np.where(tau == t + 1)
                cnf_matrix.update(idx, t, y_pred[loc].cpu(), y[loc].cpu())

        cnf_matrix.save(exp_dir, filename)
        return cnf_matrix




