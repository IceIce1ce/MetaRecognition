import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.utils import _BatchNorm
from .spectral_norm import SpectralNorm

class Baseline_Spectral_Norm(nn.Module):
    def __init__(self, num_classes):
        super(Baseline_Spectral_Norm, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec1 = SpectralNorm(self.conv1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # Layer 2
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec2 = SpectralNorm(self.conv2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Layer 3
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec3a = SpectralNorm(self.conv3a)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec3b = SpectralNorm(self.conv3b)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Layer 4
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec4a = SpectralNorm(self.conv4a)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec4b = SpectralNorm(self.conv4b)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Layer 5
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec5a = SpectralNorm(self.conv5a)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spec5b = SpectralNorm(self.conv5b)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        # out layer, should compare with and without SpectralNorm of fc layers
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv3d):
            #     torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, nn.BatchNorm3d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # use init weight below just for training from scratch
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.005)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x):
        x = self.pool1(self.relu(self.spec1(x)))
        x = self.pool2(self.relu(self.spec2(x)))
        x = self.relu(self.spec3a(x))
        x = self.relu(self.spec3b(x))
        x = self.pool3(x)
        x = self.relu(self.spec4a(x))
        x = self.relu(self.spec4b(x))
        x = self.pool4(x)
        x = self.relu(self.spec5a(x))
        x = self.relu(self.spec5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = Baseline_Spectral_Norm(num_classes=101)
    # print(model)
    output = model(X) # [1, 101]
    print(output.shape)