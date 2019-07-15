import skorch
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.utils import TeeGenerator
from skorch.utils import to_tensor
import ipdb

class forecastNet(NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step_single(self, Xi, yi, **fit_params):
        self.module_.train()        
        self.optimizer_.zero_grad()
        y_pred, aux_loss = self.infer(Xi, **fit_params)

        clf_loss = self.get_loss(y_pred, yi, X=Xi, training=True)

        loss = clf_loss + aux_loss

        self.history.record_batch('aux_loss_train', aux_loss.item())
        self.history.record_batch('clf_loss_train', clf_loss.item())

        loss.backward()

        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.module_.named_parameters()),
            X=Xi,
            y=yi
        )

        return {
            'loss': loss,
            'y_pred': y_pred,
            }
    
    def validation_step(self, Xi, yi, **fit_params):
        self.module_.eval()
        with torch.no_grad():
            y_pred, aux_loss = self.infer(Xi, **fit_params)
            clf_loss = self.get_loss(y_pred, yi, X=Xi, training=False)
            self.history.record_batch('aux_loss_valid', aux_loss.item())
            self.history.record_batch('clf_loss_valid', clf_loss.item())
            loss = aux_loss + clf_loss
            
        return {
            'loss': loss,
            'y_pred': y_pred,
            }
            
    def infer(self, x, **fit_params):
        x = to_tensor(x, device=self.device)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(**x_dict)
            
        return self.module_(x, **fit_params)
            
            
        
