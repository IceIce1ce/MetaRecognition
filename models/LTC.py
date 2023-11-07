# https://github.com/MRzzm/action-recognition-models-pytorch/blob/master/3DCNN/LTC/LTC.py
import torch.nn as nn
from torch.nn import init
import torch

class LTC(nn.Module):
    def __init__(self, num_classes):
        super(LTC, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # Layer 2
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        # Layer 3
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        # Layer 4
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        # Layer 5
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        # classifier
        self.fc6 = nn.Linear(2304, 2048) # (6144, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.9)
        self.__init_weights()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias,  0)

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112) # (1, 3, 100, 71, 71)
    model = LTC(num_classes=101)
    print(model)
    output = model(X)  # [1, 101]
    print(output.shape)