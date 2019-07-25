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

    def fit_loop(self, X , y=None, epochs=None, **fit_params):
        epochs = epochs if epochs is not None else self.max_epochs

#        dataset_train, dataset_valid = self.get_split_datasets(
#            X, y, **fit_params)

        on_epoch_kwargs = {
            'dataset_train': self.datagens_train,
            'dataset_valid': self.datagens_val,
        }
        
        for _ in range(epochs):
                self.notify('on_epoch_begin', **on_epoch_kwargs)
                train_batch_count = 0

                for T in range(self.numT):
                    for Xi, yi in self.datagens_train[T]:
                        #Xi, yi = unpack_data(data)
                        yi_res = yi
                        self.notify('on_batch_begin', X=Xi, y=yi_res, training=True)
                        step = self.train_step(Xi, yi, **fit_params)
                        self.history.record_batch('train_loss', step['loss'].item())
                        self.history.record_batch('train_batch_size', yi.shape[0])
                        self.notify('on_batch_end', X=Xi, y=yi_res, training=True, **step)
                        train_batch_count += 1
                self.history.record("train_batch_count", train_batch_count)

                valid_batch_count = 0
                for T in range(self.numT):
                    for Xi, yi in self.datagens_val[T]:
                        yi_res = yi
                        self.notify('on_batch_begin', X=Xi, y=yi_res, training=False)
                        step = self.validation_step(Xi, yi, **fit_params)
                        self.history.record_batch('valid_loss', step['loss'].item())
                        self.history.record_batch('valid_batch_size', yi.shape[0])
                        self.notify('on_batch_end', X=Xi, y=yi_res, training=False, **step)
                        valid_batch_count += 1

                self.history.record("valid_batch_count", valid_batch_count)

                self.notify('on_epoch_end', **on_epoch_kwargs)

        return self
    
    def fit(self, X, y=None, **fit_params):
        """Initialize and fit the module.
        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).
        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.
        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.
        """
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.datagens_train = fit_params.pop('datagens_train')
        self.datagens_val = fit_params.pop('datagens_val')
        self.numT = len(self.datagens_val)
        
        print('numT: ', self.numT)

        self.partial_fit(X, y, **fit_params)
        return self

    def forward_iter(self, X, training=False, device='cpu'):
        for T in range(self.numT):
            for Xi ,y in X[T]:
                yp = self.evaluation_step(Xi, training=training)
                if isinstance(yp, tuple):
                    yield tuple(n.to(device) for n in yp)
                else:
                    yield yp.to(device)

