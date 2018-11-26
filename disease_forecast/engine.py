import numpy as np
import os
import pandas as pd
from scipy.misc import imsave
from tqdm import tqdm
import torch
import ipdb
from disease_forecast import models, utils, datagen

class Model:
    def __init__(self, module_image, module_temporal, \
            module_forecast, module_task, fusion):
        # Model Architectures: Image
        model_dict = {
                'tadpole': models.Tadpole,
                'long': models.Longitudinal,
                'cov': models.Covariate,
                'lstm': models.LSTM,
                'append_time' : models.AppendTime,
                'dx': models.ANN_DX 
                #  'cnn3d': models.Sononet3d
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
        
        # Initialize the optimizer
        self.optm = torch.optim.Adam(
                #  list(self.model_image.parameters()) + \
                list(self.model_long.parameters()) + \
                list(self.model_cov.parameters()) + \
                list(self.model_temporal.parameters()) + \
                #  list(self.model_forecast.parameters()) + \
                list(self.model_task.parameters())
                )

    def train(self, datagen_train, datagen_val, exp_dir, num_epochs):
        
       # For each epoch,
       for epoch in range(num_epochs):
           self.optm.zero_grad() 

           # STEP 1: GENERATE TRAIN BATCH -------------------------------------
           # Get the training batch
           x_train_batch = next(datagen_train)
           (B, T) = x_train_batch.shape
           print('Batch size = {}, Time steps = {}'.format(B, T))

           # STEP 2: EXTRACT INPUT VALUES -------------------------------------
           # Get time data : x_time_data = (B, T)
           x_time_data = datagen.get_time_batch(x_train_batch, as_tensor=True)
           # Get image data : x_img_data: (B, T-1, Di)
           x_img_data = datagen.get_img_batch(x_train_batch, as_tensor=True) 
           # Get longitudinal data : x_long_data: (B, T-1, Dl)  
           x_long_data = datagen.get_long_batch(x_train_batch, as_tensor=True)
           # Get covariate data : x_cov_data: (B, T-1, Dc)
           x_cov_data = datagen.get_cov_batch(x_train_batch, as_tensor=True)
           print('Input data dims: Time={}, Image={}, Long={}, Cov={}'.\
                      format(x_time_data.shape, x_img_data.shape, \
                      x_long_data.shape, x_cov_data.shape))
           
           # STEP 3: EXTRACT TASK LABELS ---------------------------------------
           y_dx = datagen.get_labels(x_train_batch, task='dx', as_tensor=True)

           # STEP 3: MODULE 1: FEATURE EXTRACTION -----------------------------
           # Get image features :  x_img_feat = (B, T-1, Fi) 
           x_img_data = x_img_data.view(B*(T-1), 1, -1)
           x_img_feat = self.model_image.forward(x_img_data)
           x_img_feat = x_img_feat.view(B, T-1, -1)
           print('Image features dim = ', x_img_feat.shape)

           # Get longitudinal features : x_long_feat: (B, T-1, Fl)
           x_long_data = x_long_data.view(B*(T-1), 1, -1)
           x_long_feat = self.model_long.forward(x_long_data)
           x_long_feat = x_long_feat.view(B, T-1, -1)
           print('Longitudinal features dim = ', x_long_feat.shape)

           # Get Covariate features : x_cov_feat: (B, T-1, Fc)
           x_cov_data = x_cov_data.view(B*(T-1), 1, -1)
           x_cov_feat = self.model_cov.forward(x_cov_data)
           x_cov_feat = x_cov_feat.view(B, T-1, -1)
           print('Covariate features dim = ', x_cov_feat.shape)

           # STEP 4: MULTI MODAL FEATURE FUSION -------------------------------
           # Fuse the features
           # x_feat: (B, T-1, F_i+F_l+F_c) = (B, T-1, F)
           if self.fusion=='latefuse':
               x_feat = torch.cat((x_img_feat, x_long_feat, x_cov_feat), -1)
           print('Feature fusion dims = ', x_feat.shape)

           # STEP 5: MODULE 2: TEMPORAL FUSION --------------------------------
           # X_temp: (B, F_t)
           x_temp = self.model_temporal.forward(x_feat) if T>2 else x_feat
           print('Temporal dims = ', x_temp.shape)

           # STEP 6: MODULE 3: FORECASTING ------------------------------------
           # x_forecast: (B, F_f)
           x_forecast = self.model_forecast.forward(x_temp, x_time_data)
           print('Forecast dims = ', x_forecast.shape)

           # STEP 7: MODULE 4: TASK SPECIFIC LAYERS ---------------------------
           # DX Classification Module
           ypred = self.model_task.forward(x_forecast)
           print('Task: pred = {}, labels = {}'.format(ypred.shape, y_dx.shape))
           loss = self.model_task.loss(ypred, y_dx)
           print('Loss = ', loss)

           # STEP 8: MODEL TRAINING -------------------------------------------
           loss.backward()
           self.optm.step()

           ipdb.set_trace()
