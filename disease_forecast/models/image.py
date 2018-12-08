import torch
import torch.nn as nn
import torch.nn.functional as F

from disease_forecast.models.unet_utils import * 

class Tadpole1(nn.Module):
    def __init__(self, num_input, num_output):
        super(Tadpole1, self).__init__()
        self.affine = nn.Linear(num_input, num_output)
        self.bn = nn.BatchNorm1d(num_output)
        
    def forward(self, x):
        x = x.squeeze()
        x = self.affine(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class Tadpole2(nn.Module):
    def __init__(self):
        super(Tadpole2, self).__init__()
        self.affine1 = nn.Linear(692, 400)
        self.affine2 = nn.Linear(400, 200)
 
    def forward(self, x):
        x = x.squeeze()
        x = self.affine1(x)
        x = nn.BatchNorm1d(400)(x)
        x = F.relu(x)

        x = self.affine2(x)
        x = nn.BatchNorm1d(200)(x)
        x = F.relu(x)
        return x

class unet_3D(nn.Module):
    def __init__(self, feature_scale=1, n_classes=3, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        f = [64, 128, 256, 512, 1024]
        self.filters = [int(x / self.feature_scale) for x in f]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, self.filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(self.filters[0], self.filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3 = UnetConv3(self.filters[1], self.filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(self.filters[2], self.filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(self.filters[3], self.filters[4], self.is_batchnorm)

        # final conv (without any concat)
        
        self.globalavgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.denselayer = nn.Linear(self.filters[1],10)
        self.outputlayer = nn.Linear(10,3)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
                #m = m.double()
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                #m = m.double()

    def forward(self, inputs):
        print("before conv1")
        conv1 = self.conv1(inputs)
        print("after conv1")
        
        maxpool1 = self.maxpool1(conv1)
        print("shape maxpool1: ", maxpool1.size())

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        print("shape maxpool2: ", maxpool2.size())
        gap = self.globalavgpool(maxpool2)
        print(gap.size())
        gap = gap.view(self.filters[1])
        print("shape gap: ", gap.size())

        int1 = self.denselayer(gap)
        out = self.outputlayer(int1)

        classes = nn.Softmax(out)
        vals, indices = torch.max(classes, 0)

        return classes
