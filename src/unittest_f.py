import numpy as np
import torch

def change_in_params(params_old, params_new):
    for key in params_old:
        param_old = params_old[key]
        param_new = params_new[key]
        for i in range(len(param_old)):
            try:
                assert int(torch.eq(param_old[i], param_new[i]).all().data)==0
                        
            except:
                print("No change in the weights of module "+key+" layer "+str(i))
                pass



