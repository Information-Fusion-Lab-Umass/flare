[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/activation.py:46 return F.threshold(input, self.threshold, self.value, self.inplace)
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/activation.py:46 return F.threshold(input, self.threshold, self.value, self.inplace)
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(input and target shapes do not match: input [20], target [30] at /pytorch/aten/src/THNN/generic/MSECriterion.c:12) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/activation.py:46 return F.threshold(input, self.threshold, self.value, self.inplace)
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.00240325927734375MiB
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/activation.py:46 return F.threshold(input, self.threshold, self.value, self.inplace)
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(20,) 0.078125KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(20,) 0.078125KiB
 Total=0.00247955322265625MiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(20,) 0.078125KiB
 Total=0.00247955322265625MiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(20,) 0.078125KiB
 Total=0.00247955322265625MiB
*** RuntimeError(Expected object of type torch.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.00(0.00)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 20) 2.34375KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(20,) 0.078125KiB
 Total=0.00247955322265625MiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/activation.py:46 return F.threshold(input, self.threshold, self.value, self.inplace)
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/activation.py:46 return F.threshold(input, self.threshold, self.value, self.inplace)
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/activation.py:46 return F.threshold(input, self.threshold, self.value, self.inplace)
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
 + Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Threshold.forward:46:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Threshold.forward:46:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 Total=0.00484466552734375MiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/linear.py:55 return F.linear(input, self.weight, self.bias)
 + Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
 + Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 + Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py:1027 if bias is not None:
 + linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
*** RuntimeError(size mismatch, m1: [1 x 30], m2: [40 x 30] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:249) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1026, in linear
    output = input.matmul(weight.t())
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
  linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.0049591064453125MiB
*** RuntimeError(size mismatch, m1: [1 x 30], m2: [40 x 30] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:249) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 55, in forward
    return F.linear(input, self.weight, self.bias)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1026, in linear
    output = input.matmul(weight.t())
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
  linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.0049591064453125MiB
*** RuntimeError(size mismatch, m1: [1 x 30], m2: [40 x 30] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:249) ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 55, in forward
    return F.linear(input, self.weight, self.bias)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1026, in linear
    output = input.matmul(weight.t())
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
  linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.0049591064453125MiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/linear.py:55 return F.linear(input, self.weight, self.bias)
 + Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
 + Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
 + Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py:1027 if bias is not None:
 + linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
  linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.0049591064453125MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
  linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.0049591064453125MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
  linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.0049591064453125MiB
*** RuntimeError(Expected object of type torch.cuda.FloatTensor but found type torch.LongTensor for argument #2 'target') ***
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 421, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1716, in mse_loss
    return _pointwise_loss(lambda a, b: (a - b) ** 2, torch._C._nn.mse_loss, input, target, reduction)
  File "/cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in _pointwise_loss
    return lambd_optimized(input, target, reduction)
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
Current tensors:
 cuda:0
  Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
  Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
  linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 Total=0.0049591064453125MiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/linear.py:55 return F.linear(input, self.weight, self.bias)
 + Linear.forward:55:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 + Linear.forward:55:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
 + Linear.forward:55:0 torch.cuda.FloatTensor(40,) 0.15625KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py:1027 if bias is not None:
 + linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py:421 return F.mse_loss(input, target, reduction=self.reduction)
 + MSELoss.forward:421:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/container.py:90 for module in self._modules.values():
 + Sequential.forward:90:0 torch.cuda.FloatTensor(30, 40) 4.6875KiB
 + Sequential.forward:90:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 + Sequential.forward:90:0 torch.cuda.FloatTensor(40,) 0.15625KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/functional.py:1027 if bias is not None:
 + linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/module.py:478 for hook in self._forward_hooks.values():
 + Module.__call__:478:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
[cuda:0 alloc=0.01(0.01)MiB cache=1.00(1.00)MiB]
 /cm/shared/apps/python/3.6.6-1810/lib/python3.6/site-packages/torch/nn/modules/loss.py:421 return F.mse_loss(input, target, reduction=self.reduction)
 + MSELoss.forward:421:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
 - linear:1027:0 torch.cuda.FloatTensor(30,) 0.1171875KiB
