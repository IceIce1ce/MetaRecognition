# https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
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

class ResNet(nn.Module):
    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=101, pretrained=None, type_resnet='resnet101'):
        super().__init__()
        self.pretrained = pretrained
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7), stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # pretrained from kinetics-700
        if pretrained:
            if type_resnet == 'resnet18':
                print('Loading pretrained weights for resnet18...')
                self.__load_pretrained_weights_resnet18()
            elif type_resnet == 'resnet34':
                print('Loading pretrained weights for resnet34...')
                self.__load_pretrained_weights_resnet34()
            elif type_resnet == 'resnet50':
                print('Loading pretrained weights for resnet50...')
                self.__load_pretrained_weights_resnet50()
            elif type_resnet == 'resnet101':
                print('Loading pretrained weights for resnet101...')
                self.__load_pretrained_weights_resnet101()

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride),
                                           nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def __load_pretrained_weights_resnet18(self):
        corresp_name = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var',
                        'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight',
                        'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean',
                        'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked',
                        'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight',
                        'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias',
                        'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
                        'layer2.1.bn2.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight',
                        'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias',
                        'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean',
                        'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked',
                        'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias',
                        'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean',
                        'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var',
                        'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked']
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

    def __load_pretrained_weights_resnet34(self):
        corresp_name = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var',
                        'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight',
                        'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean',
                        'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked',
                        'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias',
                        'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
                        'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var',
                        'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight',
                        'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean',
                        'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked',
                        'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias',
                        'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var',
                        'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight',
                        'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight',
                        'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
                        'layer3.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight',
                        'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean',
                        'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked',
                        'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias',
                        'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var',
                        'layer3.5.bn1.num_batches_tracked', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer4.0.conv1.weight',
                        'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean',
                        'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var',
                        'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight',
                        'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean',
                        'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked']
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

    def __load_pretrained_weights_resnet50(self):
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
                        'layer3.5.bn3.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight',
                        'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.conv3.weight', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean',
                        'layer4.0.bn3.running_var', 'layer4.0.bn3.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var',
                        'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight',
                        'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean',
                        'layer4.1.bn3.running_var', 'layer4.1.bn3.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked',
                        'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias',
                        'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.num_batches_tracked']
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

    def __load_pretrained_weights_resnet101(self):
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
                        'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight',
                        'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean',
                        'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var', 'layer4.1.bn3.num_batches_tracked',
                        'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias',
                        'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.num_batches_tracked']
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

def Resnet3D_10(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)

def Resnet3D_18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)

def Resnet3D_34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)

def Resnet3D_50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)

def Resnet3D_101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)

def Resnet3D_152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)

def Resnet3D_200(**kwargs):
    return ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = Resnet3D_18(n_classes=101, pretrained='pretrained/resnet-18-kinetics.pth', type_resnet='resnet18')
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)