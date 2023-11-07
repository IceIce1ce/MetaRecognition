import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.utils import _BatchNorm
from .FFC3D import FFC_BN_ACT, FFCSE_block

class FFC_C3D(nn.Module):
    def __init__(self, num_classes):
        super(FFC_C3D, self).__init__()
        # Layer 1
        self.conv1 = FFC_BN_ACT(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), ratio_gin=0, ratio_gout=0.5, stride=1, enable_lfu=True, activation_layer=nn.ReLU)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # Layer 2
        self.conv2 = FFC_BN_ACT(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), ratio_gin=0.5, ratio_gout=0.5, stride=1, enable_lfu=True, activation_layer=nn.ReLU)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Layer 3
        self.conv3a = FFC_BN_ACT(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), ratio_gin=0.5, ratio_gout=0.5, stride=1, enable_lfu=True, activation_layer=nn.ReLU)
        self.conv3b = FFC_BN_ACT(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), ratio_gin=0.5, ratio_gout=0.5, stride=1, enable_lfu=True, activation_layer=nn.ReLU)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Layer 4
        self.conv4a = FFC_BN_ACT(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1), ratio_gin=0.5, ratio_gout=0.5, stride=1, enable_lfu=True, activation_layer=nn.ReLU)
        self.conv4b = FFC_BN_ACT(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1), ratio_gin=0.5, ratio_gout=0, stride=1, enable_lfu=True, activation_layer=nn.ReLU)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        # out layer
        self.fc6 = nn.Linear(65536, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.se_block = FFCSE_block(512, 0)
        self.__init_weight()

    def forward(self, x, use_se=True):
        x_l1, x_g1 = self.conv1(x)
        x_l1, x_g1 = self.pool1(x_l1), self.pool1(x_g1)
        x_l2, x_g2 = self.conv2((x_l1, x_g1))
        x_l2, x_g2 = self.pool2(x_l2), self.pool2(x_g2)
        x_l3, x_g3 = self.conv3a((x_l2, x_g2))
        x_l3, x_g3 = self.conv3b((x_l3, x_g3))
        x_l3, x_g3 = self.pool3(x_l3), self.pool3(x_g3)
        x_l4, x_g4 = self.conv4a((x_l3, x_g3))
        x_l4, x_g4 = self.conv4b((x_l4, x_g4))
        if use_se:
            x_l4, x_g4 = self.se_block((x_l4, x_g4))
        out = self.pool4(x_l4) # x_g4 = 0
        out = out.view(-1, 65536)
        out = self.relu(self.fc6(out))
        out = self.dropout(out)
        out = self.relu(self.fc7(out))
        out = self.dropout(out)
        out = self.fc8(out)
        return out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.005)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = FFC_C3D(num_classes=101)
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)