import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

net_version = {0 : [3, 196, 3,0, 2,0, 2,0, 2,0, 2,0, 1,0],  # bottle      900  450 225 112.5 56   28   14
               1 : [3, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # cable       1024 512 256 128   64   32   16
               2 : [3, 225, 2,0, 2,0, 3,0, 2,0, 1,0, 2,0],  # capsule     1000 500 250 125   62.5 31   15.5
               3 : [3, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # carpet      1024 512 256 128   64   32   16
               4 : [1, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # grid        1024 512 256 128   64   32   16
               5 : [3, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # hazelnut    1024 512 256 128   64   32   16
               6 : [3, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # leather     1024 512 256 128   64   32   16
               7 : [3, 100, 5,0, 1,0, 2,0, 2,0, 2,0, 3,0],  # metal_nut   700  350 175 87.5  43.5 21.5 10.5
               8 : [3, 144, 4,0, 2,0, 1,0, 2,0, 2,0, 2,0],  # pill        800  400 200 100   50   25   12.5
               9 : [1, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # screw       1024 512 256 128   64   32   16
               10 :[3, 169, 4,0, 1,0, 2,0, 2,0, 1,0, 2,0],  # tile        840  420 210 105   52.5 26   13
               11 :[3, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # toothbrush  1024 512 256 128   64   32   16
               12 :[3, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # transistor  1024 512 256 128   64   32   16
               13 :[3, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0],  # wood        1024 512 256 128   64   32   16
               14 :[1, 256, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0]}  # zipper      1024 512 256 128   64   32   16


class MVTecAD_LeNet_ELU(BaseNet):

    def __init__(self, version=5):
        super().__init__()

        self.rep_dim = 512
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(net_version[version][0], 4, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(4, 5, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(5, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(5, 12, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(12, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(12, 13, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(13, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(13, 84, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(84, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(84, 85, 5, bias=False, padding=2)
        self.bn2d6 = nn.BatchNorm2d(85, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(85 * net_version[version][1], self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.elu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.elu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(F.elu(self.bn2d6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MVTecAD_LeNet_ELU_Autoencoder(BaseNet):

    def __init__(self, version=5):
        super().__init__()

        self.rep_dim = 512
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(net_version[version][0], 4, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn2d1 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        
        self.conv2 = nn.Conv2d(4, 5, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2d2 = nn.BatchNorm2d(5, eps=1e-04, affine=False)
        
        self.conv3 = nn.Conv2d(5, 12, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn2d3 = nn.BatchNorm2d(12, eps=1e-04, affine=False)
        
        self.conv4 = nn.Conv2d(12, 13, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn2d4 = nn.BatchNorm2d(13, eps=1e-04, affine=False)
        
        self.conv5 = nn.Conv2d(13, 84, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.bn2d5 = nn.BatchNorm2d(84, eps=1e-04, affine=False)
        
        self.conv6 = nn.Conv2d(84, 85, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv6.weight)
        self.bn2d6 = nn.BatchNorm2d(85, eps=1e-04, affine=False)
        
        self.fc1 = nn.Linear(85 * net_version[version][1], self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (16 * 16)), 85, 5, bias=False, padding=net_version[version][2], output_padding=net_version[version][3])
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.bn2d7 = nn.BatchNorm2d(85, eps=1e-04, affine=False)
        
        self.deconv2 = nn.ConvTranspose2d(85, 84, 5, bias=False, padding=net_version[version][4], output_padding=net_version[version][5])
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.bn2d8 = nn.BatchNorm2d(84, eps=1e-04, affine=False)
        
        self.deconv3 = nn.ConvTranspose2d(84, 13, 5, bias=False, padding=net_version[version][6], output_padding=net_version[version][7])
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.bn2d9 = nn.BatchNorm2d(13, eps=1e-04, affine=False)
        
        self.deconv4 = nn.ConvTranspose2d(13, 12, 5, bias=False, padding=net_version[version][8], output_padding=net_version[version][9])
        nn.init.xavier_uniform_(self.deconv4.weight)
        self.bn2d10 = nn.BatchNorm2d(12, eps=1e-04, affine=False)
        
        self.deconv5 = nn.ConvTranspose2d(12, 5, 5, bias=False, padding=net_version[version][10], output_padding=net_version[version][11])
        nn.init.xavier_uniform_(self.deconv5.weight)
        self.bn2d11 = nn.BatchNorm2d(5, eps=1e-04, affine=False)
        
        self.deconv6 = nn.ConvTranspose2d(5, 4, 5, bias=False, padding=net_version[version][12], output_padding=net_version[version][13])
        nn.init.xavier_uniform_(self.deconv6.weight)
        self.bn2d12 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        
        self.deconv7 = nn.ConvTranspose2d(4, net_version[version][0], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv7.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.elu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.elu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(F.elu(self.bn2d6(x)))
        
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        
        x = x.view(x.size(0), int(self.rep_dim / (16 * 16)), 16, 16)
        x = F.elu(x)
        
        x = self.deconv1(x) # 16
        x = F.interpolate(F.elu(self.bn2d7(x)), scale_factor=2)
        x = self.deconv2(x) # 32
        x = F.interpolate(F.elu(self.bn2d8(x)), scale_factor=2)
        x = self.deconv3(x) # 64
        x = F.interpolate(F.elu(self.bn2d9(x)), scale_factor=2)
        x = self.deconv4(x) # 128
        x = F.interpolate(F.elu(self.bn2d10(x)), scale_factor=2)
        x = self.deconv5(x) # 256
        x = F.interpolate(F.elu(self.bn2d11(x)), scale_factor=2)
        x = self.deconv6(x) # 512
        x = F.interpolate(F.elu(self.bn2d12(x)), scale_factor=2)
        x = self.deconv7(x) # 1024
        
        x = torch.sigmoid(x)
        return x
