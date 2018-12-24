from torch import nn
import torch
import sys
import threading
import profile

model = nn.Sequential(nn.Linear(40, 30), nn.ReLU()).cuda()
criterion = nn.MSELoss().cuda()

memory_profiler = profile.CUDAMemoryProfiler(
    [model, criterion],
    filename='cuda_memory.profile'
)

sys.settrace(memory_profiler)
threading.settrace(memory_profiler)

inp = torch.randn(40).cuda()
out = model(inp)
print("Input size: ",inp.size())
print("Out size: ",out.size())
print("Arange size: ",torch.arange(1,31).size())
loss = criterion(out, torch.arange(1, 31).float().cuda())
