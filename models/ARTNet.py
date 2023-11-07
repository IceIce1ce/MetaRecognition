# https://github.com/MRzzm/action-recognition-models-pytorch/blob/master/3DCNN/ARTNet/ARTNet.py
import torch.nn as nn
import torch

class SMART_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(SMART_block, self).__init__()
        # appearance branch
        self.appearance_conv = nn.Conv3d(in_channel, out_channel, kernel_size=(1, kernel_size[1], kernel_size[2]), stride=stride, padding=(0, padding[1], padding[2]), bias=False)
        self.appearance_bn = nn.BatchNorm3d(out_channel)
        # relation branch
        self.relation_conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relation_bn1 = nn.BatchNorm3d(out_channel)
        self.relation_pooling = nn.Conv3d(out_channel, out_channel//2, kernel_size=1, stride=1, groups=out_channel//2, bias=False)
        self.relation_bn2 = nn.BatchNorm3d(out_channel//2)
        # nn.init.constant_(self.relation_pooling.weight, 0.5)
        # self.relation_pooling.weight.requires_grad = False
        # smart block
        self.reduce = nn.Conv3d(out_channel + out_channel//2, out_channel, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        if in_channel != out_channel or stride[0] != 1 or stride[1] != 1:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(out_channel))
        else:
            self.down_sample = None
        self.__init_weight() # check again for large validation loss

    def forward(self, x):
        residual = x
        appearance = x
        relation = x
        appearance = self.appearance_conv(appearance)
        appearance = self.appearance_bn(appearance)
        relation = self.relation_conv(relation)
        relation = self.relation_bn1(relation)
        relation = torch.pow(relation, 2) # square
        relation = self.relation_pooling(relation) # cross channel pooling
        relation = self.relation_bn2(relation)
        stream = self.relu(torch.cat([appearance, relation], 1))
        stream = self.reduce(stream)
        stream = self.reduce_bn(stream)
        if self.down_sample is not None:
            residual = self.down_sample(x)
        stream += residual
        stream = self.relu(stream)
        return stream

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ARTNet(nn.Module):
    def __init__(self, num_classes):
        super(ARTNet, self).__init__()
        self.conv1 = SMART_block(3, 64, kernel_size=(3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3))
        self.conv2 = nn.Sequential(SMART_block(64, 64),
                                   SMART_block(64, 64))
        self.conv3 = nn.Sequential(SMART_block(64, 128, stride=(2, 2, 2)),
                                   SMART_block(128, 128))
        self.conv4 = nn.Sequential(SMART_block(128, 256, stride=(2, 2, 2)),
                                   SMART_block(256, 256))
        self.conv5 = nn.Sequential(SMART_block(256, 512, stride=(2, 2, 2)),
                                   SMART_block(512, 512))
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 7, 7))
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = ARTNet(num_classes=101)
    print(model)
    output = model(X)  # [1, 101]
    print(output.shape)