# https://github.com/okankop/Efficient-3DCNNs/blob/master/models/resnext.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from collections import OrderedDict

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out

class ResNeXtBottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', cardinality=32, num_classes=400, pretrained=None, type_resnext='resnext101'):
        super(ResNeXt, self).__init__()
        self.pretrained = pretrained
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # pretrained from kinetics-600
        if pretrained:
            if type_resnext == 'resnext101':
                print('Loading pretrained weights for resnext101...')
                self.__load_pretrained_weights_resnext101()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                                           nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def __load_pretrained_weights_resnext101(self):
        corresp_name = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var',
                        'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.0.conv3.weight',
                        'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias',
                        'layer1.0.downsample.1.running_mean', 'layer1.0.downsample.1.running_var', 'layer1.0.downsample.1.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean',
                        'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked',
                        'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias',
                        'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var',
                        'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked', 'layer2.0.conv1.weight',
                        'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean',
                        'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked',
                        'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight',
                        'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean',
                        'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked',
                        'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias',
                        'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight', 'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var',
                        'layer2.2.bn3.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight',
                        'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked', 'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean',
                        'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked',
                        'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias',
                        'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean',
                        'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var',
                        'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight',
                        'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean',
                        'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked',
                        'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
                        'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
                        'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked', 'layer3.4.conv1.weight',
                        'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean',
                        'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.4.conv3.weight', 'layer3.4.bn3.weight', 'layer3.4.bn3.bias', 'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var', 'layer3.4.bn3.num_batches_tracked',
                        'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.bn1.num_batches_tracked', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias',
                        'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer3.5.conv3.weight', 'layer3.5.bn3.weight', 'layer3.5.bn3.bias', 'layer3.5.bn3.running_mean', 'layer3.5.bn3.running_var',
                        'layer3.5.bn3.num_batches_tracked', 'layer3.6.conv1.weight', 'layer3.6.bn1.weight', 'layer3.6.bn1.bias', 'layer3.6.bn1.running_mean', 'layer3.6.bn1.running_var', 'layer3.6.bn1.num_batches_tracked', 'layer3.6.conv2.weight',
                        'layer3.6.bn2.weight', 'layer3.6.bn2.bias', 'layer3.6.bn2.running_mean', 'layer3.6.bn2.running_var', 'layer3.6.bn2.num_batches_tracked', 'layer3.6.conv3.weight', 'layer3.6.bn3.weight', 'layer3.6.bn3.bias', 'layer3.6.bn3.running_mean',
                        'layer3.6.bn3.running_var', 'layer3.6.bn3.num_batches_tracked', 'layer3.7.conv1.weight', 'layer3.7.bn1.weight', 'layer3.7.bn1.bias', 'layer3.7.bn1.running_mean', 'layer3.7.bn1.running_var', 'layer3.7.bn1.num_batches_tracked',
                        'layer3.7.conv2.weight', 'layer3.7.bn2.weight', 'layer3.7.bn2.bias', 'layer3.7.bn2.running_mean', 'layer3.7.bn2.running_var', 'layer3.7.bn2.num_batches_tracked', 'layer3.7.conv3.weight', 'layer3.7.bn3.weight', 'layer3.7.bn3.bias',
                        'layer3.7.bn3.running_mean', 'layer3.7.bn3.running_var', 'layer3.7.bn3.num_batches_tracked', 'layer3.8.conv1.weight', 'layer3.8.bn1.weight', 'layer3.8.bn1.bias', 'layer3.8.bn1.running_mean', 'layer3.8.bn1.running_var',
                        'layer3.8.bn1.num_batches_tracked', 'layer3.8.conv2.weight', 'layer3.8.bn2.weight', 'layer3.8.bn2.bias', 'layer3.8.bn2.running_mean', 'layer3.8.bn2.running_var', 'layer3.8.bn2.num_batches_tracked', 'layer3.8.conv3.weight',
                        'layer3.8.bn3.weight', 'layer3.8.bn3.bias', 'layer3.8.bn3.running_mean', 'layer3.8.bn3.running_var', 'layer3.8.bn3.num_batches_tracked', 'layer3.9.conv1.weight', 'layer3.9.bn1.weight', 'layer3.9.bn1.bias', 'layer3.9.bn1.running_mean',
                        'layer3.9.bn1.running_var', 'layer3.9.bn1.num_batches_tracked', 'layer3.9.conv2.weight', 'layer3.9.bn2.weight', 'layer3.9.bn2.bias', 'layer3.9.bn2.running_mean', 'layer3.9.bn2.running_var', 'layer3.9.bn2.num_batches_tracked',
                        'layer3.9.conv3.weight', 'layer3.9.bn3.weight', 'layer3.9.bn3.bias', 'layer3.9.bn3.running_mean', 'layer3.9.bn3.running_var', 'layer3.9.bn3.num_batches_tracked', 'layer3.10.conv1.weight', 'layer3.10.bn1.weight', 'layer3.10.bn1.bias',
                        'layer3.10.bn1.running_mean', 'layer3.10.bn1.running_var', 'layer3.10.bn1.num_batches_tracked', 'layer3.10.conv2.weight', 'layer3.10.bn2.weight', 'layer3.10.bn2.bias', 'layer3.10.bn2.running_mean', 'layer3.10.bn2.running_var',
                        'layer3.10.bn2.num_batches_tracked', 'layer3.10.conv3.weight', 'layer3.10.bn3.weight', 'layer3.10.bn3.bias', 'layer3.10.bn3.running_mean', 'layer3.10.bn3.running_var', 'layer3.10.bn3.num_batches_tracked', 'layer3.11.conv1.weight',
                        'layer3.11.bn1.weight', 'layer3.11.bn1.bias', 'layer3.11.bn1.running_mean', 'layer3.11.bn1.running_var', 'layer3.11.bn1.num_batches_tracked', 'layer3.11.conv2.weight', 'layer3.11.bn2.weight', 'layer3.11.bn2.bias',
                        'layer3.11.bn2.running_mean', 'layer3.11.bn2.running_var', 'layer3.11.bn2.num_batches_tracked', 'layer3.11.conv3.weight', 'layer3.11.bn3.weight', 'layer3.11.bn3.bias', 'layer3.11.bn3.running_mean', 'layer3.11.bn3.running_var',
                        'layer3.11.bn3.num_batches_tracked', 'layer3.12.conv1.weight', 'layer3.12.bn1.weight', 'layer3.12.bn1.bias', 'layer3.12.bn1.running_mean', 'layer3.12.bn1.running_var', 'layer3.12.bn1.num_batches_tracked', 'layer3.12.conv2.weight',
                        'layer3.12.bn2.weight', 'layer3.12.bn2.bias', 'layer3.12.bn2.running_mean', 'layer3.12.bn2.running_var', 'layer3.12.bn2.num_batches_tracked', 'layer3.12.conv3.weight', 'layer3.12.bn3.weight', 'layer3.12.bn3.bias',
                        'layer3.12.bn3.running_mean', 'layer3.12.bn3.running_var', 'layer3.12.bn3.num_batches_tracked', 'layer3.13.conv1.weight', 'layer3.13.bn1.weight', 'layer3.13.bn1.bias', 'layer3.13.bn1.running_mean', 'layer3.13.bn1.running_var',
                        'layer3.13.bn1.num_batches_tracked', 'layer3.13.conv2.weight', 'layer3.13.bn2.weight', 'layer3.13.bn2.bias', 'layer3.13.bn2.running_mean', 'layer3.13.bn2.running_var', 'layer3.13.bn2.num_batches_tracked', 'layer3.13.conv3.weight',
                        'layer3.13.bn3.weight', 'layer3.13.bn3.bias', 'layer3.13.bn3.running_mean', 'layer3.13.bn3.running_var', 'layer3.13.bn3.num_batches_tracked', 'layer3.14.conv1.weight', 'layer3.14.bn1.weight', 'layer3.14.bn1.bias',
                        'layer3.14.bn1.running_mean', 'layer3.14.bn1.running_var', 'layer3.14.bn1.num_batches_tracked', 'layer3.14.conv2.weight', 'layer3.14.bn2.weight', 'layer3.14.bn2.bias', 'layer3.14.bn2.running_mean', 'layer3.14.bn2.running_var',
                        'layer3.14.bn2.num_batches_tracked', 'layer3.14.conv3.weight', 'layer3.14.bn3.weight', 'layer3.14.bn3.bias', 'layer3.14.bn3.running_mean', 'layer3.14.bn3.running_var', 'layer3.14.bn3.num_batches_tracked', 'layer3.15.conv1.weight',
                        'layer3.15.bn1.weight', 'layer3.15.bn1.bias', 'layer3.15.bn1.running_mean', 'layer3.15.bn1.running_var', 'layer3.15.bn1.num_batches_tracked', 'layer3.15.conv2.weight', 'layer3.15.bn2.weight', 'layer3.15.bn2.bias',
                        'layer3.15.bn2.running_mean', 'layer3.15.bn2.running_var', 'layer3.15.bn2.num_batches_tracked', 'layer3.15.conv3.weight', 'layer3.15.bn3.weight', 'layer3.15.bn3.bias', 'layer3.15.bn3.running_mean', 'layer3.15.bn3.running_var',
                        'layer3.15.bn3.num_batches_tracked', 'layer3.16.conv1.weight', 'layer3.16.bn1.weight', 'layer3.16.bn1.bias', 'layer3.16.bn1.running_mean', 'layer3.16.bn1.running_var', 'layer3.16.bn1.num_batches_tracked', 'layer3.16.conv2.weight',
                        'layer3.16.bn2.weight', 'layer3.16.bn2.bias', 'layer3.16.bn2.running_mean', 'layer3.16.bn2.running_var', 'layer3.16.bn2.num_batches_tracked', 'layer3.16.conv3.weight', 'layer3.16.bn3.weight', 'layer3.16.bn3.bias',
                        'layer3.16.bn3.running_mean', 'layer3.16.bn3.running_var', 'layer3.16.bn3.num_batches_tracked', 'layer3.17.conv1.weight', 'layer3.17.bn1.weight', 'layer3.17.bn1.bias', 'layer3.17.bn1.running_mean', 'layer3.17.bn1.running_var',
                        'layer3.17.bn1.num_batches_tracked', 'layer3.17.conv2.weight', 'layer3.17.bn2.weight', 'layer3.17.bn2.bias', 'layer3.17.bn2.running_mean', 'layer3.17.bn2.running_var', 'layer3.17.bn2.num_batches_tracked', 'layer3.17.conv3.weight',
                        'layer3.17.bn3.weight', 'layer3.17.bn3.bias', 'layer3.17.bn3.running_mean', 'layer3.17.bn3.running_var', 'layer3.17.bn3.num_batches_tracked', 'layer3.18.conv1.weight', 'layer3.18.bn1.weight', 'layer3.18.bn1.bias',
                        'layer3.18.bn1.running_mean', 'layer3.18.bn1.running_var', 'layer3.18.bn1.num_batches_tracked', 'layer3.18.conv2.weight', 'layer3.18.bn2.weight', 'layer3.18.bn2.bias', 'layer3.18.bn2.running_mean', 'layer3.18.bn2.running_var',
                        'layer3.18.bn2.num_batches_tracked', 'layer3.18.conv3.weight', 'layer3.18.bn3.weight', 'layer3.18.bn3.bias', 'layer3.18.bn3.running_mean', 'layer3.18.bn3.running_var', 'layer3.18.bn3.num_batches_tracked', 'layer3.19.conv1.weight',
                        'layer3.19.bn1.weight', 'layer3.19.bn1.bias', 'layer3.19.bn1.running_mean', 'layer3.19.bn1.running_var', 'layer3.19.bn1.num_batches_tracked', 'layer3.19.conv2.weight', 'layer3.19.bn2.weight', 'layer3.19.bn2.bias',
                        'layer3.19.bn2.running_mean', 'layer3.19.bn2.running_var', 'layer3.19.bn2.num_batches_tracked', 'layer3.19.conv3.weight', 'layer3.19.bn3.weight', 'layer3.19.bn3.bias', 'layer3.19.bn3.running_mean', 'layer3.19.bn3.running_var',
                        'layer3.19.bn3.num_batches_tracked', 'layer3.20.conv1.weight', 'layer3.20.bn1.weight', 'layer3.20.bn1.bias', 'layer3.20.bn1.running_mean', 'layer3.20.bn1.running_var', 'layer3.20.bn1.num_batches_tracked', 'layer3.20.conv2.weight',
                        'layer3.20.bn2.weight', 'layer3.20.bn2.bias', 'layer3.20.bn2.running_mean', 'layer3.20.bn2.running_var', 'layer3.20.bn2.num_batches_tracked', 'layer3.20.conv3.weight', 'layer3.20.bn3.weight', 'layer3.20.bn3.bias',
                        'layer3.20.bn3.running_mean', 'layer3.20.bn3.running_var', 'layer3.20.bn3.num_batches_tracked', 'layer3.21.conv1.weight', 'layer3.21.bn1.weight', 'layer3.21.bn1.bias', 'layer3.21.bn1.running_mean', 'layer3.21.bn1.running_var',
                        'layer3.21.bn1.num_batches_tracked', 'layer3.21.conv2.weight', 'layer3.21.bn2.weight', 'layer3.21.bn2.bias', 'layer3.21.bn2.running_mean', 'layer3.21.bn2.running_var', 'layer3.21.bn2.num_batches_tracked', 'layer3.21.conv3.weight',
                        'layer3.21.bn3.weight', 'layer3.21.bn3.bias', 'layer3.21.bn3.running_mean', 'layer3.21.bn3.running_var', 'layer3.21.bn3.num_batches_tracked', 'layer3.22.conv1.weight', 'layer3.22.bn1.weight', 'layer3.22.bn1.bias',
                        'layer3.22.bn1.running_mean', 'layer3.22.bn1.running_var', 'layer3.22.bn1.num_batches_tracked', 'layer3.22.conv2.weight', 'layer3.22.bn2.weight', 'layer3.22.bn2.bias', 'layer3.22.bn2.running_mean', 'layer3.22.bn2.running_var',
                        'layer3.22.bn2.num_batches_tracked', 'layer3.22.conv3.weight', 'layer3.22.bn3.weight', 'layer3.22.bn3.bias', 'layer3.22.bn3.running_mean', 'layer3.22.bn3.running_var', 'layer3.22.bn3.num_batches_tracked', 'layer4.0.conv1.weight',
                        'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean',
                        'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.conv3.weight', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var', 'layer4.0.bn3.num_batches_tracked',
                        'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked',
                        'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias',
                        'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var',
                        'layer4.1.bn3.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.weight',
                        'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean',
                        'layer4.2.bn3.running_var', 'layer4.2.bn3.num_batches_tracked']
        p_dict = torch.load(self.pretrained)
        s_dict = self.state_dict()
        new_p_dict = OrderedDict()
        for k, v in p_dict['state_dict'].items():
            k = k.replace('module.', '') # rename .module from parallel setting to general setting
            new_p_dict[k] = v
        for name in new_p_dict:
            if name not in corresp_name:
                continue
            s_dict[name] = new_p_dict[name]
        self.load_state_dict(s_dict)

def Resnext3D_50(**kwargs):
    return ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)

def Resnext3D_101(**kwargs):
    return ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)

def Resnext3D_152(**kwargs):
    return ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = Resnext3D_101(sample_size=112, sample_duration=16, num_classes=101, pretrained='pretrained/pretrained_resnext3D/kinetics_resnext_101_RGB_16_best.pth', type_resnext='resnext101')
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)