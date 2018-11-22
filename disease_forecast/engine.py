import numpy as np
import os
import pandas as pd
from scipy.misc import imsave
from tqdm import tqdm
#  import matplotlib.pyplot as plt

#  import tensorflow as tf
#  from keras.callbacks import ModelCheckpoint, CSVLogger
#  from keras.losses import categorical_crossentropy
#  import keras.backend as K

from disease_forecast import models, utils, datagen

class Model:
    def __init__(self, module_image, module_temporal, module_forecast, fusion):
        # Model Architectures: Image
        model_dict_image = {
                'Tadpole': models.Tadpole,
                'CNN3D_sononet': models.Sononet3d
                }
        
        # Load model: image architecture
        model_image_name = module_image.pop('name')
        self.model_image = model_dict[model_image_name](**module_image)
        
        # Load model: temporal architecture
        model_temporal_name = module_temporal.pop('name')
        self.model_temporal = model_dict[model_temporal_name](**module_temporal)
        
        # Load model: forecast architecture
        model_forecast_name = module_forecast.pop('name')
        self.model_forecast = model_dict_image[model_forecast_name](**module_forecast)

        self.fusion = fusion

    def train(self, datagen_train, datagen_val):

       # For each epoch,
       for epoch in num_epochs:

           #  x_train_batch = next(datagen_train)
           x_train_batch = datagen.get_Batch(patients, B, n_t, feat_flag) 

           # Get image features
           # x_img_feat: (B, T, F_i)
           x_img_data, T = datagen.get_img_batch(x_train_batch) 
           x_img_feat = []
           for t in range(T):
               x_img_feat.append(self.model_image.forward(x_img_data))
           x_img_feat = np.array(x_img_feat)

           # Get longitudinal features
           # x_long_feat: (B, T, F_l)
           x_long_data,_ = datagen.get_long_batch(x_train_batch)
           x_cov_data,_ = datagen.get_covariate_batch(x_train_batch)
           x_long_feat = []
           for t in range(T):
               x_long_feat.append(self.model_long.forward(x_long_data))
           x_long_feat = np.array(x_long_feat)

           # Get covariate features
           # x_cov_feat: (B, T, F_c)
           x_cov_data, _ = datagen.get_long_batch(x_train_batch)
           x_cov_feat = []
           for t in range(T):
               x_cov_feat.append(self.model_covariate.forward(x_cov_data))
           x_cov_feat = np.array(x_cov_feat) 

           # Fuse the features
           if self.fusion=='latefuse':
               x_feat = np.concatenate((x_img_feat, x_long_feat, x_cov_feat), axis=-1)
           #  elif self.fusion=='shortfuse':
           #      x_feat = models.shortfuse(x_img_feat, x_long_feat, x_cov_feat)

           # Temporal Module




    #  def train(self, datagen_train, datagen_val, output_dir, ckpt_period, fit_params)

'''        
class Model:
    #  def __init__(self, name, trained_model, compile_params, model_params):
        # Model architectures
        model_dict = {
                'DenseNet3D': models.DenseNet3D,
                'UNet2D': models.UNet2D,
                'DilatedDenseNet': models.DilatedDenseNet
                }

        # Load model architecture
        self.model = model_dict[name](**model_params)
        #  print(self.model.summary())
        
        # Load saved weights
        if trained_model!='None':
            self.model.load_weights(trained_model)

        # Compile the model
        self.model.compile(
                optimizer=compile_params['optimizer'],
                loss=self.loss(compile_params['loss']),
                metrics=compile_params['metrics'])

    def loss(self, loss_name):
        if loss_name=='categorical_crossentropy':
            loss = categorical_crossentropy
        elif loss_name=='dice_coef':
            def dice_coef(y_true, y_pred, smooth=1):
                y_true_f = K.flatten(y_true)
                y_pred_f = K.flatten(y_pred)
                intersection = K.sum(y_true_f * y_pred_f)
                dice_coef = (2. * intersection + smooth) /\
                        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
                return 1 - dice_coef
            loss = dice_coef
        return loss

    def train(self, datagen_train, datagen_val, output_dir, ckpt_period, fit_params):
        # Callbacks
        wt_path = output_dir+'/checkpoints/'+'wt-{epoch:02d}-{val_acc:.2f}.h5'
        wt_saver = ModelCheckpoint(wt_path, monitor='val_acc', verbose=1,
                                     save_weights_only=True, period=ckpt_period)
        log_path = output_dir+'/logs/log.csv'
        csv_logger = CSVLogger(log_path, separator=',', append=False)
        callbacks = [wt_saver, csv_logger]
        
        # Train the model
        self.model.fit_generator(generator=datagen_train, 
                validation_data=datagen_val, callbacks=callbacks,
                verbose=2, **fit_params)

        #  Save the plots
        #  self.save_plots(log_path, output_dir+'/logs/')
    #
    #  def save_plots(self, log_path, output_dir):
    #      # Save the plots
    #      logs = pd.read_csv(log_path)
    #      acc_train, loss_train = logs['acc'], logs['loss']
    #      acc_val, loss_val = logs['val_acc'], logs['val_loss']
    #      # Plot accuracy
    #      plt.figure()
    #      plt.plot(acc_train, c='b', label='Training')
    #      plt.plot(acc_val, c='g', label='Validation')
    #      plt.title('Accuracy'); plt.legend();
    #      plt.savefig(output_dir+'accuracy.png', dpi=300); plt.close()
    #      # Plot loss
    #      plt.figure()
    #      plt.plot(loss_train, c='b', label='Training')
    #      plt.plot(loss_val, c='g', label='Validation')
    #      plt.title('Loss'); plt.legend();
    #      plt.savefig(output_dir+'loss.png', dpi=300); plt.close()
    #
    
    def test(self, output_dir, test_dir, trained_model, img_dims):
        
        self.model.load_weights(trained_model)
        img_dims[-1] = 1
        tasks = [x for x in os.listdir(test_dir) if 'Task' in x]
        # Iterate over tasks
        for task in tasks:
            print(task)
            x_path = os.path.join(test_dir, task, 'imagesTr')
            y_path = os.path.join(test_dir, task, 'labelsTr_npz')
            save_path = os.path.join(output_dir, task)
            utils.create_dirs([save_path])
            
            # Iterate over all images in task
            num_subclasses=-1
            for name in os.listdir(x_path):
                # Image
                img = utils.load_img(os.path.join(x_path, name))
                img = utils.resize_img(img, img_dims)
                
                # Annotation
                annot_path = os.path.join(y_path, name[:-3]+'npz')
                annot = utils.load_img(annot_path)
                #if num_subclasses==-1:
                #    num_subclasses = int(np.max(annot)+1)
                max_annot = int(np.max(annot))

                output = np.zeros_like(img)
                for sub_label in range(1, max_annot+1):
                    sub_label_value = utils.get_sublabel_value(
                            sub_label, annot_path)
                    cond_map = np.ones_like(img)*sub_label_value/255.
                    img_cond = np.concatenate((img, cond_map), axis=-1)
                    model_output = self.model.predict(img_cond[np.newaxis,...])
                    print(sub_label, model_output[...,0].min(), model_output[...,0].max(), model_output[...,1].min(), model_output[...,1].max())
                    model_output = np.argmax(model_output, axis=-1)[0]
                    output[np.where(model_output==1)] = sub_label
                output = np.squeeze(output).astype(int)
                annot = utils.resize_img(annot, img_dims)

                #  inputs = np.array(inputs)
                #  model_outputs = self.model.predict(inputs)
                #  outputs = np.squeeze(np.argmax(model_outputs, axis=-1))*sub_label
                #
                #  if output.max()!=0:
                #      output = np.round(255*output/output.max()).astype(int)
                #  else:
                #      output = output.astype(int)
                #  if annot.max()!=0:
                #      annot = np.squeeze(np.round(255*annot/annot.max()).astype(int))
                #  else:
                #      annot = np.squeeze(annot.astype(int))
                #  img = np.squeeze(np.round(255*img).astype(int))
                #  concat_img = np.concatenate((img, annot, np.squeeze(output)), axis=1)
#  imsave(os.path.join(save_path, name), concat_img)
'''
