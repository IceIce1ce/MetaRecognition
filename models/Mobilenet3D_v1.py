# https://github.com/okankop/Efficient-3DCNNs/blob/master/models/mobilenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False), nn.BatchNorm3d(oup), nn.ReLU(inplace=True))

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet3D_v1(nn.Module):
    def __init__(self, num_classes=600, width_mult=1., pretrained=None):
        super(MobileNet3D_v1, self).__init__()
        self.pretrained = pretrained
        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [[64,   1, (2, 2, 2)], [128,  2, (2, 2, 2)], [256,  2, (2, 2, 2)], [512,  6, (2, 2, 2)], [1024, 2, (1, 1, 1)],]
        self.features = [conv_bn(3, input_channel, (1, 2, 2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_channel, num_classes))
        if pretrained:
            if width_mult == 0.5:
                print('Loading pretrained weights for mobilenet3D_v1 with width = 0.5...')
                self.__load_pretrained_weights_mobilenet3d_v1_05()
            elif width_mult == 1.0:
                print('Loading pretrained weights for mobilenet3D_v1 with width = 1.0...')
                self.__load_pretrained_weights_mobilenet3d_v1_10()
            elif width_mult == 1.5:
                print('Loading pretrained weights for mobilenet3D_v1 with width = 1.5...')
                self.__load_pretrained_weights_mobilenet3d_v1_15()
            elif width_mult == 2.0:
                print('Loading pretrained weights for mobilenet3D_v1 with width = 2.0...')
                self.__load_pretrained_weights_mobilenet3d_v1_20()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def __load_pretrained_weights_mobilenet3d_v1_05(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv1.weight', 'features.1.bn1.weight', 'features.1.bn1.bias',
                        'features.1.bn1.running_mean', 'features.1.bn1.running_var', 'features.1.bn1.num_batches_tracked', 'features.1.conv2.weight', 'features.1.bn2.weight', 'features.1.bn2.bias', 'features.1.bn2.running_mean', 'features.1.bn2.running_var',
                        'features.1.bn2.num_batches_tracked', 'features.2.conv1.weight', 'features.2.bn1.weight', 'features.2.bn1.bias', 'features.2.bn1.running_mean', 'features.2.bn1.running_var', 'features.2.bn1.num_batches_tracked', 'features.2.conv2.weight',
                        'features.2.bn2.weight', 'features.2.bn2.bias', 'features.2.bn2.running_mean', 'features.2.bn2.running_var', 'features.2.bn2.num_batches_tracked', 'features.3.conv1.weight', 'features.3.bn1.weight', 'features.3.bn1.bias',
                        'features.3.bn1.running_mean', 'features.3.bn1.running_var', 'features.3.bn1.num_batches_tracked', 'features.3.conv2.weight', 'features.3.bn2.weight', 'features.3.bn2.bias', 'features.3.bn2.running_mean', 'features.3.bn2.running_var',
                        'features.3.bn2.num_batches_tracked', 'features.4.conv1.weight', 'features.4.bn1.weight', 'features.4.bn1.bias', 'features.4.bn1.running_mean', 'features.4.bn1.running_var', 'features.4.bn1.num_batches_tracked', 'features.4.conv2.weight',
                        'features.4.bn2.weight', 'features.4.bn2.bias', 'features.4.bn2.running_mean', 'features.4.bn2.running_var', 'features.4.bn2.num_batches_tracked', 'features.5.conv1.weight', 'features.5.bn1.weight', 'features.5.bn1.bias',
                        'features.5.bn1.running_mean', 'features.5.bn1.running_var', 'features.5.bn1.num_batches_tracked', 'features.5.conv2.weight', 'features.5.bn2.weight', 'features.5.bn2.bias', 'features.5.bn2.running_mean', 'features.5.bn2.running_var',
                        'features.5.bn2.num_batches_tracked', 'features.6.conv1.weight', 'features.6.bn1.weight', 'features.6.bn1.bias', 'features.6.bn1.running_mean', 'features.6.bn1.running_var', 'features.6.bn1.num_batches_tracked', 'features.6.conv2.weight',
                        'features.6.bn2.weight', 'features.6.bn2.bias', 'features.6.bn2.running_mean', 'features.6.bn2.running_var', 'features.6.bn2.num_batches_tracked', 'features.7.conv1.weight', 'features.7.bn1.weight', 'features.7.bn1.bias',
                        'features.7.bn1.running_mean', 'features.7.bn1.running_var', 'features.7.bn1.num_batches_tracked', 'features.7.conv2.weight', 'features.7.bn2.weight', 'features.7.bn2.bias', 'features.7.bn2.running_mean', 'features.7.bn2.running_var',
                        'features.7.bn2.num_batches_tracked', 'features.8.conv1.weight', 'features.8.bn1.weight', 'features.8.bn1.bias', 'features.8.bn1.running_mean', 'features.8.bn1.running_var', 'features.8.bn1.num_batches_tracked', 'features.8.conv2.weight',
                        'features.8.bn2.weight', 'features.8.bn2.bias', 'features.8.bn2.running_mean', 'features.8.bn2.running_var', 'features.8.bn2.num_batches_tracked', 'features.9.conv1.weight', 'features.9.bn1.weight', 'features.9.bn1.bias',
                        'features.9.bn1.running_mean', 'features.9.bn1.running_var', 'features.9.bn1.num_batches_tracked', 'features.9.conv2.weight', 'features.9.bn2.weight', 'features.9.bn2.bias', 'features.9.bn2.running_mean', 'features.9.bn2.running_var',
                        'features.9.bn2.num_batches_tracked', 'features.10.conv1.weight', 'features.10.bn1.weight', 'features.10.bn1.bias', 'features.10.bn1.running_mean', 'features.10.bn1.running_var', 'features.10.bn1.num_batches_tracked',
                        'features.10.conv2.weight', 'features.10.bn2.weight', 'features.10.bn2.bias', 'features.10.bn2.running_mean', 'features.10.bn2.running_var', 'features.10.bn2.num_batches_tracked', 'features.11.conv1.weight', 'features.11.bn1.weight',
                        'features.11.bn1.bias', 'features.11.bn1.running_mean', 'features.11.bn1.running_var', 'features.11.bn1.num_batches_tracked', 'features.11.conv2.weight', 'features.11.bn2.weight', 'features.11.bn2.bias', 'features.11.bn2.running_mean',
                        'features.11.bn2.running_var', 'features.11.bn2.num_batches_tracked', 'features.12.conv1.weight', 'features.12.bn1.weight', 'features.12.bn1.bias', 'features.12.bn1.running_mean', 'features.12.bn1.running_var',
                        'features.12.bn1.num_batches_tracked', 'features.12.conv2.weight', 'features.12.bn2.weight', 'features.12.bn2.bias', 'features.12.bn2.running_mean', 'features.12.bn2.running_var', 'features.12.bn2.num_batches_tracked',
                        'features.13.conv1.weight', 'features.13.bn1.weight', 'features.13.bn1.bias', 'features.13.bn1.running_mean', 'features.13.bn1.running_var', 'features.13.bn1.num_batches_tracked', 'features.13.conv2.weight', 'features.13.bn2.weight',
                        'features.13.bn2.bias', 'features.13.bn2.running_mean', 'features.13.bn2.running_var', 'features.13.bn2.num_batches_tracked']
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

    def __load_pretrained_weights_mobilenet3d_v1_10(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv1.weight', 'features.1.bn1.weight', 'features.1.bn1.bias',
                        'features.1.bn1.running_mean', 'features.1.bn1.running_var', 'features.1.bn1.num_batches_tracked', 'features.1.conv2.weight', 'features.1.bn2.weight', 'features.1.bn2.bias', 'features.1.bn2.running_mean', 'features.1.bn2.running_var',
                        'features.1.bn2.num_batches_tracked', 'features.2.conv1.weight', 'features.2.bn1.weight', 'features.2.bn1.bias', 'features.2.bn1.running_mean', 'features.2.bn1.running_var', 'features.2.bn1.num_batches_tracked', 'features.2.conv2.weight',
                        'features.2.bn2.weight', 'features.2.bn2.bias', 'features.2.bn2.running_mean', 'features.2.bn2.running_var', 'features.2.bn2.num_batches_tracked', 'features.3.conv1.weight', 'features.3.bn1.weight', 'features.3.bn1.bias',
                        'features.3.bn1.running_mean', 'features.3.bn1.running_var', 'features.3.bn1.num_batches_tracked', 'features.3.conv2.weight', 'features.3.bn2.weight', 'features.3.bn2.bias', 'features.3.bn2.running_mean', 'features.3.bn2.running_var',
                        'features.3.bn2.num_batches_tracked', 'features.4.conv1.weight', 'features.4.bn1.weight', 'features.4.bn1.bias', 'features.4.bn1.running_mean', 'features.4.bn1.running_var', 'features.4.bn1.num_batches_tracked', 'features.4.conv2.weight',
                        'features.4.bn2.weight', 'features.4.bn2.bias', 'features.4.bn2.running_mean', 'features.4.bn2.running_var', 'features.4.bn2.num_batches_tracked', 'features.5.conv1.weight', 'features.5.bn1.weight', 'features.5.bn1.bias',
                        'features.5.bn1.running_mean', 'features.5.bn1.running_var', 'features.5.bn1.num_batches_tracked', 'features.5.conv2.weight', 'features.5.bn2.weight', 'features.5.bn2.bias', 'features.5.bn2.running_mean', 'features.5.bn2.running_var',
                        'features.5.bn2.num_batches_tracked', 'features.6.conv1.weight', 'features.6.bn1.weight', 'features.6.bn1.bias', 'features.6.bn1.running_mean', 'features.6.bn1.running_var', 'features.6.bn1.num_batches_tracked', 'features.6.conv2.weight',
                        'features.6.bn2.weight', 'features.6.bn2.bias', 'features.6.bn2.running_mean', 'features.6.bn2.running_var', 'features.6.bn2.num_batches_tracked', 'features.7.conv1.weight', 'features.7.bn1.weight', 'features.7.bn1.bias',
                        'features.7.bn1.running_mean', 'features.7.bn1.running_var', 'features.7.bn1.num_batches_tracked', 'features.7.conv2.weight', 'features.7.bn2.weight', 'features.7.bn2.bias', 'features.7.bn2.running_mean', 'features.7.bn2.running_var',
                        'features.7.bn2.num_batches_tracked', 'features.8.conv1.weight', 'features.8.bn1.weight', 'features.8.bn1.bias', 'features.8.bn1.running_mean', 'features.8.bn1.running_var', 'features.8.bn1.num_batches_tracked', 'features.8.conv2.weight',
                        'features.8.bn2.weight', 'features.8.bn2.bias', 'features.8.bn2.running_mean', 'features.8.bn2.running_var', 'features.8.bn2.num_batches_tracked', 'features.9.conv1.weight', 'features.9.bn1.weight', 'features.9.bn1.bias',
                        'features.9.bn1.running_mean', 'features.9.bn1.running_var', 'features.9.bn1.num_batches_tracked', 'features.9.conv2.weight', 'features.9.bn2.weight', 'features.9.bn2.bias', 'features.9.bn2.running_mean', 'features.9.bn2.running_var',
                        'features.9.bn2.num_batches_tracked', 'features.10.conv1.weight', 'features.10.bn1.weight', 'features.10.bn1.bias', 'features.10.bn1.running_mean', 'features.10.bn1.running_var', 'features.10.bn1.num_batches_tracked',
                        'features.10.conv2.weight', 'features.10.bn2.weight', 'features.10.bn2.bias', 'features.10.bn2.running_mean', 'features.10.bn2.running_var', 'features.10.bn2.num_batches_tracked', 'features.11.conv1.weight', 'features.11.bn1.weight',
                        'features.11.bn1.bias', 'features.11.bn1.running_mean', 'features.11.bn1.running_var', 'features.11.bn1.num_batches_tracked', 'features.11.conv2.weight', 'features.11.bn2.weight', 'features.11.bn2.bias', 'features.11.bn2.running_mean',
                        'features.11.bn2.running_var', 'features.11.bn2.num_batches_tracked', 'features.12.conv1.weight', 'features.12.bn1.weight', 'features.12.bn1.bias', 'features.12.bn1.running_mean', 'features.12.bn1.running_var',
                        'features.12.bn1.num_batches_tracked', 'features.12.conv2.weight', 'features.12.bn2.weight', 'features.12.bn2.bias', 'features.12.bn2.running_mean', 'features.12.bn2.running_var', 'features.12.bn2.num_batches_tracked',
                        'features.13.conv1.weight', 'features.13.bn1.weight', 'features.13.bn1.bias', 'features.13.bn1.running_mean', 'features.13.bn1.running_var', 'features.13.bn1.num_batches_tracked', 'features.13.conv2.weight', 'features.13.bn2.weight',
                        'features.13.bn2.bias', 'features.13.bn2.running_mean', 'features.13.bn2.running_var', 'features.13.bn2.num_batches_tracked']
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

    def __load_pretrained_weights_mobilenet3d_v1_15(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv1.weight', 'features.1.bn1.weight', 'features.1.bn1.bias',
                        'features.1.bn1.running_mean', 'features.1.bn1.running_var', 'features.1.bn1.num_batches_tracked', 'features.1.conv2.weight', 'features.1.bn2.weight', 'features.1.bn2.bias', 'features.1.bn2.running_mean', 'features.1.bn2.running_var',
                        'features.1.bn2.num_batches_tracked', 'features.2.conv1.weight', 'features.2.bn1.weight', 'features.2.bn1.bias', 'features.2.bn1.running_mean', 'features.2.bn1.running_var', 'features.2.bn1.num_batches_tracked', 'features.2.conv2.weight',
                        'features.2.bn2.weight', 'features.2.bn2.bias', 'features.2.bn2.running_mean', 'features.2.bn2.running_var', 'features.2.bn2.num_batches_tracked', 'features.3.conv1.weight', 'features.3.bn1.weight', 'features.3.bn1.bias',
                        'features.3.bn1.running_mean', 'features.3.bn1.running_var', 'features.3.bn1.num_batches_tracked', 'features.3.conv2.weight', 'features.3.bn2.weight', 'features.3.bn2.bias', 'features.3.bn2.running_mean', 'features.3.bn2.running_var',
                        'features.3.bn2.num_batches_tracked', 'features.4.conv1.weight', 'features.4.bn1.weight', 'features.4.bn1.bias', 'features.4.bn1.running_mean', 'features.4.bn1.running_var', 'features.4.bn1.num_batches_tracked', 'features.4.conv2.weight',
                        'features.4.bn2.weight', 'features.4.bn2.bias', 'features.4.bn2.running_mean', 'features.4.bn2.running_var', 'features.4.bn2.num_batches_tracked', 'features.5.conv1.weight', 'features.5.bn1.weight', 'features.5.bn1.bias',
                        'features.5.bn1.running_mean', 'features.5.bn1.running_var', 'features.5.bn1.num_batches_tracked', 'features.5.conv2.weight', 'features.5.bn2.weight', 'features.5.bn2.bias', 'features.5.bn2.running_mean', 'features.5.bn2.running_var',
                        'features.5.bn2.num_batches_tracked', 'features.6.conv1.weight', 'features.6.bn1.weight', 'features.6.bn1.bias', 'features.6.bn1.running_mean', 'features.6.bn1.running_var', 'features.6.bn1.num_batches_tracked', 'features.6.conv2.weight',
                        'features.6.bn2.weight', 'features.6.bn2.bias', 'features.6.bn2.running_mean', 'features.6.bn2.running_var', 'features.6.bn2.num_batches_tracked', 'features.7.conv1.weight', 'features.7.bn1.weight', 'features.7.bn1.bias',
                        'features.7.bn1.running_mean', 'features.7.bn1.running_var', 'features.7.bn1.num_batches_tracked', 'features.7.conv2.weight', 'features.7.bn2.weight', 'features.7.bn2.bias', 'features.7.bn2.running_mean', 'features.7.bn2.running_var',
                        'features.7.bn2.num_batches_tracked', 'features.8.conv1.weight', 'features.8.bn1.weight', 'features.8.bn1.bias', 'features.8.bn1.running_mean', 'features.8.bn1.running_var', 'features.8.bn1.num_batches_tracked', 'features.8.conv2.weight',
                        'features.8.bn2.weight', 'features.8.bn2.bias', 'features.8.bn2.running_mean', 'features.8.bn2.running_var', 'features.8.bn2.num_batches_tracked', 'features.9.conv1.weight', 'features.9.bn1.weight', 'features.9.bn1.bias',
                        'features.9.bn1.running_mean', 'features.9.bn1.running_var', 'features.9.bn1.num_batches_tracked', 'features.9.conv2.weight', 'features.9.bn2.weight', 'features.9.bn2.bias', 'features.9.bn2.running_mean', 'features.9.bn2.running_var',
                        'features.9.bn2.num_batches_tracked', 'features.10.conv1.weight', 'features.10.bn1.weight', 'features.10.bn1.bias', 'features.10.bn1.running_mean', 'features.10.bn1.running_var', 'features.10.bn1.num_batches_tracked',
                        'features.10.conv2.weight', 'features.10.bn2.weight', 'features.10.bn2.bias', 'features.10.bn2.running_mean', 'features.10.bn2.running_var', 'features.10.bn2.num_batches_tracked', 'features.11.conv1.weight', 'features.11.bn1.weight',
                        'features.11.bn1.bias', 'features.11.bn1.running_mean', 'features.11.bn1.running_var', 'features.11.bn1.num_batches_tracked', 'features.11.conv2.weight', 'features.11.bn2.weight', 'features.11.bn2.bias', 'features.11.bn2.running_mean',
                        'features.11.bn2.running_var', 'features.11.bn2.num_batches_tracked', 'features.12.conv1.weight', 'features.12.bn1.weight', 'features.12.bn1.bias', 'features.12.bn1.running_mean', 'features.12.bn1.running_var',
                        'features.12.bn1.num_batches_tracked', 'features.12.conv2.weight', 'features.12.bn2.weight', 'features.12.bn2.bias', 'features.12.bn2.running_mean', 'features.12.bn2.running_var', 'features.12.bn2.num_batches_tracked',
                        'features.13.conv1.weight', 'features.13.bn1.weight', 'features.13.bn1.bias', 'features.13.bn1.running_mean', 'features.13.bn1.running_var', 'features.13.bn1.num_batches_tracked', 'features.13.conv2.weight', 'features.13.bn2.weight',
                        'features.13.bn2.bias', 'features.13.bn2.running_mean', 'features.13.bn2.running_var', 'features.13.bn2.num_batches_tracked']
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

    def __load_pretrained_weights_mobilenet3d_v1_20(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv1.weight', 'features.1.bn1.weight', 'features.1.bn1.bias',
                        'features.1.bn1.running_mean', 'features.1.bn1.running_var', 'features.1.bn1.num_batches_tracked', 'features.1.conv2.weight', 'features.1.bn2.weight', 'features.1.bn2.bias', 'features.1.bn2.running_mean', 'features.1.bn2.running_var',
                        'features.1.bn2.num_batches_tracked', 'features.2.conv1.weight', 'features.2.bn1.weight', 'features.2.bn1.bias', 'features.2.bn1.running_mean', 'features.2.bn1.running_var', 'features.2.bn1.num_batches_tracked', 'features.2.conv2.weight',
                        'features.2.bn2.weight', 'features.2.bn2.bias', 'features.2.bn2.running_mean', 'features.2.bn2.running_var', 'features.2.bn2.num_batches_tracked', 'features.3.conv1.weight', 'features.3.bn1.weight', 'features.3.bn1.bias',
                        'features.3.bn1.running_mean', 'features.3.bn1.running_var', 'features.3.bn1.num_batches_tracked', 'features.3.conv2.weight', 'features.3.bn2.weight', 'features.3.bn2.bias', 'features.3.bn2.running_mean', 'features.3.bn2.running_var',
                        'features.3.bn2.num_batches_tracked', 'features.4.conv1.weight', 'features.4.bn1.weight', 'features.4.bn1.bias', 'features.4.bn1.running_mean', 'features.4.bn1.running_var', 'features.4.bn1.num_batches_tracked', 'features.4.conv2.weight',
                        'features.4.bn2.weight', 'features.4.bn2.bias', 'features.4.bn2.running_mean', 'features.4.bn2.running_var', 'features.4.bn2.num_batches_tracked', 'features.5.conv1.weight', 'features.5.bn1.weight', 'features.5.bn1.bias',
                        'features.5.bn1.running_mean', 'features.5.bn1.running_var', 'features.5.bn1.num_batches_tracked', 'features.5.conv2.weight', 'features.5.bn2.weight', 'features.5.bn2.bias', 'features.5.bn2.running_mean', 'features.5.bn2.running_var',
                        'features.5.bn2.num_batches_tracked', 'features.6.conv1.weight', 'features.6.bn1.weight', 'features.6.bn1.bias', 'features.6.bn1.running_mean', 'features.6.bn1.running_var', 'features.6.bn1.num_batches_tracked', 'features.6.conv2.weight',
                        'features.6.bn2.weight', 'features.6.bn2.bias', 'features.6.bn2.running_mean', 'features.6.bn2.running_var', 'features.6.bn2.num_batches_tracked', 'features.7.conv1.weight', 'features.7.bn1.weight', 'features.7.bn1.bias',
                        'features.7.bn1.running_mean', 'features.7.bn1.running_var', 'features.7.bn1.num_batches_tracked', 'features.7.conv2.weight', 'features.7.bn2.weight', 'features.7.bn2.bias', 'features.7.bn2.running_mean', 'features.7.bn2.running_var',
                        'features.7.bn2.num_batches_tracked', 'features.8.conv1.weight', 'features.8.bn1.weight', 'features.8.bn1.bias', 'features.8.bn1.running_mean', 'features.8.bn1.running_var', 'features.8.bn1.num_batches_tracked', 'features.8.conv2.weight',
                        'features.8.bn2.weight', 'features.8.bn2.bias', 'features.8.bn2.running_mean', 'features.8.bn2.running_var', 'features.8.bn2.num_batches_tracked', 'features.9.conv1.weight', 'features.9.bn1.weight', 'features.9.bn1.bias',
                        'features.9.bn1.running_mean', 'features.9.bn1.running_var', 'features.9.bn1.num_batches_tracked', 'features.9.conv2.weight', 'features.9.bn2.weight', 'features.9.bn2.bias', 'features.9.bn2.running_mean', 'features.9.bn2.running_var',
                        'features.9.bn2.num_batches_tracked', 'features.10.conv1.weight', 'features.10.bn1.weight', 'features.10.bn1.bias', 'features.10.bn1.running_mean', 'features.10.bn1.running_var', 'features.10.bn1.num_batches_tracked',
                        'features.10.conv2.weight', 'features.10.bn2.weight', 'features.10.bn2.bias', 'features.10.bn2.running_mean', 'features.10.bn2.running_var', 'features.10.bn2.num_batches_tracked', 'features.11.conv1.weight', 'features.11.bn1.weight',
                        'features.11.bn1.bias', 'features.11.bn1.running_mean', 'features.11.bn1.running_var', 'features.11.bn1.num_batches_tracked', 'features.11.conv2.weight', 'features.11.bn2.weight', 'features.11.bn2.bias', 'features.11.bn2.running_mean',
                        'features.11.bn2.running_var', 'features.11.bn2.num_batches_tracked', 'features.12.conv1.weight', 'features.12.bn1.weight', 'features.12.bn1.bias', 'features.12.bn1.running_mean', 'features.12.bn1.running_var',
                        'features.12.bn1.num_batches_tracked', 'features.12.conv2.weight', 'features.12.bn2.weight', 'features.12.bn2.bias', 'features.12.bn2.running_mean', 'features.12.bn2.running_var', 'features.12.bn2.num_batches_tracked',
                        'features.13.conv1.weight', 'features.13.bn1.weight', 'features.13.bn1.bias', 'features.13.bn1.running_mean', 'features.13.bn1.running_var', 'features.13.bn1.num_batches_tracked', 'features.13.conv2.weight', 'features.13.bn2.weight',
                        'features.13.bn2.bias', 'features.13.bn2.running_mean', 'features.13.bn2.running_var', 'features.13.bn2.num_batches_tracked']
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

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = MobileNet3D_v1(num_classes=101, width_mult=1.0, pretrained='pretrained/pretrained_mobilenet3D_v1/kinetics_mobilenet_1.0x_RGB_16_best.pth')
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)