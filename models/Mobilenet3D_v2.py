# https://github.com/okankop/Efficient-3DCNNs/blob/master/models/mobilenetv2.py
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False), nn.BatchNorm3d(oup), nn.ReLU6(inplace=True))

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv3d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup), nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), # dw
                                      nn.BatchNorm3d(hidden_dim),
                                      nn.ReLU6(inplace=True),
                                      # pw-linear
                                      nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                                      nn.BatchNorm3d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False), # pw
                                      nn.BatchNorm3d(hidden_dim),
                                      nn.ReLU6(inplace=True),
                                      # dw
                                      nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                                      nn.BatchNorm3d(hidden_dim),
                                      nn.ReLU6(inplace=True),
                                      # pw-linear
                                      nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                                      nn.BatchNorm3d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNet3D_v2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., pretrained=None):
        super(MobileNet3D_v2, self).__init__()
        self.pretrained = pretrained
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1,  16, 1, (1, 1, 1)], [6,  24, 2, (2, 2, 2)], [6,  32, 3, (2, 2, 2)], [6,  64, 4, (2, 2, 2)],
                                        [6,  96, 3, (1, 1, 1)], [6, 160, 3, (2, 2, 2)], [6, 320, 1, (1, 1, 1)]]
        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, (1, 2, 2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        # building classifier
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
        self._initialize_weights()
        if pretrained:
            if width_mult == 0.2:
                print('Loading pretrained weights for mobilenet3D_v2 with width = 0.2...')
                self.__load_pretrained_weights_mobilenet3d_v2_02()
            elif width_mult == 0.7:
                print('Loading pretrained weights for mobilenet3D_v2 with width = 0.7...')
                self.__load_pretrained_weights_mobilenet3d_v2_07()
            elif width_mult == 0.45:
                print('Loading pretrained weights for mobilenet3D_v2 with width = 0.45...')
                self.__load_pretrained_weights_mobilenet3d_v2_045()
            elif width_mult == 1.0:
                print('Loading pretrained weights for mobilenet3D_v2 with width = 1.0...')
                self.__load_pretrained_weights_mobilenet3d_v2_10()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def __load_pretrained_weights_mobilenet3d_v2_02(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv.0.weight', 'features.1.conv.1.weight',
                        'features.1.conv.1.bias', 'features.1.conv.1.running_mean', 'features.1.conv.1.running_var', 'features.1.conv.1.num_batches_tracked', 'features.1.conv.3.weight', 'features.1.conv.4.weight', 'features.1.conv.4.bias',
                        'features.1.conv.4.running_mean', 'features.1.conv.4.running_var', 'features.1.conv.4.num_batches_tracked', 'features.2.conv.0.weight', 'features.2.conv.1.weight', 'features.2.conv.1.bias', 'features.2.conv.1.running_mean',
                        'features.2.conv.1.running_var', 'features.2.conv.1.num_batches_tracked', 'features.2.conv.3.weight', 'features.2.conv.4.weight', 'features.2.conv.4.bias', 'features.2.conv.4.running_mean', 'features.2.conv.4.running_var',
                        'features.2.conv.4.num_batches_tracked', 'features.2.conv.6.weight', 'features.2.conv.7.weight', 'features.2.conv.7.bias', 'features.2.conv.7.running_mean', 'features.2.conv.7.running_var', 'features.2.conv.7.num_batches_tracked',
                        'features.3.conv.0.weight', 'features.3.conv.1.weight', 'features.3.conv.1.bias', 'features.3.conv.1.running_mean', 'features.3.conv.1.running_var', 'features.3.conv.1.num_batches_tracked', 'features.3.conv.3.weight',
                        'features.3.conv.4.weight', 'features.3.conv.4.bias', 'features.3.conv.4.running_mean', 'features.3.conv.4.running_var', 'features.3.conv.4.num_batches_tracked', 'features.3.conv.6.weight', 'features.3.conv.7.weight',
                        'features.3.conv.7.bias', 'features.3.conv.7.running_mean', 'features.3.conv.7.running_var', 'features.3.conv.7.num_batches_tracked', 'features.4.conv.0.weight', 'features.4.conv.1.weight', 'features.4.conv.1.bias',
                        'features.4.conv.1.running_mean', 'features.4.conv.1.running_var', 'features.4.conv.1.num_batches_tracked', 'features.4.conv.3.weight', 'features.4.conv.4.weight', 'features.4.conv.4.bias', 'features.4.conv.4.running_mean',
                        'features.4.conv.4.running_var', 'features.4.conv.4.num_batches_tracked', 'features.4.conv.6.weight', 'features.4.conv.7.weight', 'features.4.conv.7.bias', 'features.4.conv.7.running_mean', 'features.4.conv.7.running_var',
                        'features.4.conv.7.num_batches_tracked', 'features.5.conv.0.weight', 'features.5.conv.1.weight', 'features.5.conv.1.bias', 'features.5.conv.1.running_mean', 'features.5.conv.1.running_var', 'features.5.conv.1.num_batches_tracked',
                        'features.5.conv.3.weight', 'features.5.conv.4.weight', 'features.5.conv.4.bias', 'features.5.conv.4.running_mean', 'features.5.conv.4.running_var', 'features.5.conv.4.num_batches_tracked', 'features.5.conv.6.weight',
                        'features.5.conv.7.weight', 'features.5.conv.7.bias', 'features.5.conv.7.running_mean', 'features.5.conv.7.running_var', 'features.5.conv.7.num_batches_tracked', 'features.6.conv.0.weight', 'features.6.conv.1.weight',
                        'features.6.conv.1.bias', 'features.6.conv.1.running_mean', 'features.6.conv.1.running_var', 'features.6.conv.1.num_batches_tracked', 'features.6.conv.3.weight', 'features.6.conv.4.weight', 'features.6.conv.4.bias',
                        'features.6.conv.4.running_mean', 'features.6.conv.4.running_var', 'features.6.conv.4.num_batches_tracked', 'features.6.conv.6.weight', 'features.6.conv.7.weight', 'features.6.conv.7.bias', 'features.6.conv.7.running_mean',
                        'features.6.conv.7.running_var', 'features.6.conv.7.num_batches_tracked', 'features.7.conv.0.weight', 'features.7.conv.1.weight', 'features.7.conv.1.bias', 'features.7.conv.1.running_mean', 'features.7.conv.1.running_var',
                        'features.7.conv.1.num_batches_tracked', 'features.7.conv.3.weight', 'features.7.conv.4.weight', 'features.7.conv.4.bias', 'features.7.conv.4.running_mean', 'features.7.conv.4.running_var', 'features.7.conv.4.num_batches_tracked',
                        'features.7.conv.6.weight', 'features.7.conv.7.weight', 'features.7.conv.7.bias', 'features.7.conv.7.running_mean', 'features.7.conv.7.running_var', 'features.7.conv.7.num_batches_tracked', 'features.8.conv.0.weight',
                        'features.8.conv.1.weight', 'features.8.conv.1.bias', 'features.8.conv.1.running_mean', 'features.8.conv.1.running_var', 'features.8.conv.1.num_batches_tracked', 'features.8.conv.3.weight', 'features.8.conv.4.weight',
                        'features.8.conv.4.bias', 'features.8.conv.4.running_mean', 'features.8.conv.4.running_var', 'features.8.conv.4.num_batches_tracked', 'features.8.conv.6.weight', 'features.8.conv.7.weight', 'features.8.conv.7.bias',
                        'features.8.conv.7.running_mean', 'features.8.conv.7.running_var', 'features.8.conv.7.num_batches_tracked', 'features.9.conv.0.weight', 'features.9.conv.1.weight', 'features.9.conv.1.bias', 'features.9.conv.1.running_mean',
                        'features.9.conv.1.running_var', 'features.9.conv.1.num_batches_tracked', 'features.9.conv.3.weight', 'features.9.conv.4.weight', 'features.9.conv.4.bias', 'features.9.conv.4.running_mean', 'features.9.conv.4.running_var',
                        'features.9.conv.4.num_batches_tracked', 'features.9.conv.6.weight', 'features.9.conv.7.weight', 'features.9.conv.7.bias', 'features.9.conv.7.running_mean', 'features.9.conv.7.running_var', 'features.9.conv.7.num_batches_tracked',
                        'features.10.conv.0.weight', 'features.10.conv.1.weight', 'features.10.conv.1.bias', 'features.10.conv.1.running_mean', 'features.10.conv.1.running_var', 'features.10.conv.1.num_batches_tracked', 'features.10.conv.3.weight',
                        'features.10.conv.4.weight', 'features.10.conv.4.bias', 'features.10.conv.4.running_mean', 'features.10.conv.4.running_var', 'features.10.conv.4.num_batches_tracked', 'features.10.conv.6.weight', 'features.10.conv.7.weight',
                        'features.10.conv.7.bias', 'features.10.conv.7.running_mean', 'features.10.conv.7.running_var', 'features.10.conv.7.num_batches_tracked', 'features.11.conv.0.weight', 'features.11.conv.1.weight', 'features.11.conv.1.bias',
                        'features.11.conv.1.running_mean', 'features.11.conv.1.running_var', 'features.11.conv.1.num_batches_tracked', 'features.11.conv.3.weight', 'features.11.conv.4.weight', 'features.11.conv.4.bias', 'features.11.conv.4.running_mean',
                        'features.11.conv.4.running_var', 'features.11.conv.4.num_batches_tracked', 'features.11.conv.6.weight', 'features.11.conv.7.weight', 'features.11.conv.7.bias', 'features.11.conv.7.running_mean', 'features.11.conv.7.running_var',
                        'features.11.conv.7.num_batches_tracked', 'features.12.conv.0.weight', 'features.12.conv.1.weight', 'features.12.conv.1.bias', 'features.12.conv.1.running_mean', 'features.12.conv.1.running_var', 'features.12.conv.1.num_batches_tracked',
                        'features.12.conv.3.weight', 'features.12.conv.4.weight', 'features.12.conv.4.bias', 'features.12.conv.4.running_mean', 'features.12.conv.4.running_var', 'features.12.conv.4.num_batches_tracked', 'features.12.conv.6.weight',
                        'features.12.conv.7.weight', 'features.12.conv.7.bias', 'features.12.conv.7.running_mean', 'features.12.conv.7.running_var', 'features.12.conv.7.num_batches_tracked', 'features.13.conv.0.weight', 'features.13.conv.1.weight',
                        'features.13.conv.1.bias', 'features.13.conv.1.running_mean', 'features.13.conv.1.running_var', 'features.13.conv.1.num_batches_tracked', 'features.13.conv.3.weight', 'features.13.conv.4.weight', 'features.13.conv.4.bias',
                        'features.13.conv.4.running_mean', 'features.13.conv.4.running_var', 'features.13.conv.4.num_batches_tracked', 'features.13.conv.6.weight', 'features.13.conv.7.weight', 'features.13.conv.7.bias', 'features.13.conv.7.running_mean',
                        'features.13.conv.7.running_var', 'features.13.conv.7.num_batches_tracked', 'features.14.conv.0.weight', 'features.14.conv.1.weight', 'features.14.conv.1.bias', 'features.14.conv.1.running_mean', 'features.14.conv.1.running_var',
                        'features.14.conv.1.num_batches_tracked', 'features.14.conv.3.weight', 'features.14.conv.4.weight', 'features.14.conv.4.bias', 'features.14.conv.4.running_mean', 'features.14.conv.4.running_var', 'features.14.conv.4.num_batches_tracked',
                        'features.14.conv.6.weight', 'features.14.conv.7.weight', 'features.14.conv.7.bias', 'features.14.conv.7.running_mean', 'features.14.conv.7.running_var', 'features.14.conv.7.num_batches_tracked', 'features.15.conv.0.weight',
                        'features.15.conv.1.weight', 'features.15.conv.1.bias', 'features.15.conv.1.running_mean', 'features.15.conv.1.running_var', 'features.15.conv.1.num_batches_tracked', 'features.15.conv.3.weight', 'features.15.conv.4.weight',
                        'features.15.conv.4.bias', 'features.15.conv.4.running_mean', 'features.15.conv.4.running_var', 'features.15.conv.4.num_batches_tracked', 'features.15.conv.6.weight', 'features.15.conv.7.weight', 'features.15.conv.7.bias',
                        'features.15.conv.7.running_mean', 'features.15.conv.7.running_var', 'features.15.conv.7.num_batches_tracked', 'features.16.conv.0.weight', 'features.16.conv.1.weight', 'features.16.conv.1.bias', 'features.16.conv.1.running_mean',
                        'features.16.conv.1.running_var', 'features.16.conv.1.num_batches_tracked', 'features.16.conv.3.weight', 'features.16.conv.4.weight', 'features.16.conv.4.bias', 'features.16.conv.4.running_mean', 'features.16.conv.4.running_var',
                        'features.16.conv.4.num_batches_tracked', 'features.16.conv.6.weight', 'features.16.conv.7.weight', 'features.16.conv.7.bias', 'features.16.conv.7.running_mean', 'features.16.conv.7.running_var', 'features.16.conv.7.num_batches_tracked',
                        'features.17.conv.0.weight', 'features.17.conv.1.weight', 'features.17.conv.1.bias', 'features.17.conv.1.running_mean', 'features.17.conv.1.running_var', 'features.17.conv.1.num_batches_tracked', 'features.17.conv.3.weight',
                        'features.17.conv.4.weight', 'features.17.conv.4.bias', 'features.17.conv.4.running_mean', 'features.17.conv.4.running_var', 'features.17.conv.4.num_batches_tracked', 'features.17.conv.6.weight', 'features.17.conv.7.weight',
                        'features.17.conv.7.bias', 'features.17.conv.7.running_mean', 'features.17.conv.7.running_var', 'features.17.conv.7.num_batches_tracked', 'features.18.0.weight', 'features.18.1.weight', 'features.18.1.bias', 'features.18.1.running_mean',
                        'features.18.1.running_var', 'features.18.1.num_batches_tracked']
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

    def __load_pretrained_weights_mobilenet3d_v2_07(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv.0.weight', 'features.1.conv.1.weight', 'features.1.conv.1.bias',
                        'features.1.conv.1.running_mean', 'features.1.conv.1.running_var', 'features.1.conv.1.num_batches_tracked', 'features.1.conv.3.weight', 'features.1.conv.4.weight', 'features.1.conv.4.bias', 'features.1.conv.4.running_mean',
                        'features.1.conv.4.running_var', 'features.1.conv.4.num_batches_tracked', 'features.2.conv.0.weight', 'features.2.conv.1.weight', 'features.2.conv.1.bias', 'features.2.conv.1.running_mean', 'features.2.conv.1.running_var',
                        'features.2.conv.1.num_batches_tracked', 'features.2.conv.3.weight', 'features.2.conv.4.weight', 'features.2.conv.4.bias', 'features.2.conv.4.running_mean', 'features.2.conv.4.running_var', 'features.2.conv.4.num_batches_tracked',
                        'features.2.conv.6.weight', 'features.2.conv.7.weight', 'features.2.conv.7.bias', 'features.2.conv.7.running_mean', 'features.2.conv.7.running_var', 'features.2.conv.7.num_batches_tracked', 'features.3.conv.0.weight',
                        'features.3.conv.1.weight', 'features.3.conv.1.bias', 'features.3.conv.1.running_mean', 'features.3.conv.1.running_var', 'features.3.conv.1.num_batches_tracked', 'features.3.conv.3.weight', 'features.3.conv.4.weight',
                        'features.3.conv.4.bias', 'features.3.conv.4.running_mean', 'features.3.conv.4.running_var', 'features.3.conv.4.num_batches_tracked', 'features.3.conv.6.weight', 'features.3.conv.7.weight', 'features.3.conv.7.bias',
                        'features.3.conv.7.running_mean', 'features.3.conv.7.running_var', 'features.3.conv.7.num_batches_tracked', 'features.4.conv.0.weight', 'features.4.conv.1.weight', 'features.4.conv.1.bias', 'features.4.conv.1.running_mean',
                        'features.4.conv.1.running_var', 'features.4.conv.1.num_batches_tracked', 'features.4.conv.3.weight', 'features.4.conv.4.weight', 'features.4.conv.4.bias', 'features.4.conv.4.running_mean', 'features.4.conv.4.running_var',
                        'features.4.conv.4.num_batches_tracked', 'features.4.conv.6.weight', 'features.4.conv.7.weight', 'features.4.conv.7.bias', 'features.4.conv.7.running_mean', 'features.4.conv.7.running_var', 'features.4.conv.7.num_batches_tracked',
                        'features.5.conv.0.weight', 'features.5.conv.1.weight', 'features.5.conv.1.bias', 'features.5.conv.1.running_mean', 'features.5.conv.1.running_var', 'features.5.conv.1.num_batches_tracked', 'features.5.conv.3.weight',
                        'features.5.conv.4.weight', 'features.5.conv.4.bias', 'features.5.conv.4.running_mean', 'features.5.conv.4.running_var', 'features.5.conv.4.num_batches_tracked', 'features.5.conv.6.weight', 'features.5.conv.7.weight',
                        'features.5.conv.7.bias', 'features.5.conv.7.running_mean', 'features.5.conv.7.running_var', 'features.5.conv.7.num_batches_tracked', 'features.6.conv.0.weight', 'features.6.conv.1.weight', 'features.6.conv.1.bias',
                        'features.6.conv.1.running_mean', 'features.6.conv.1.running_var', 'features.6.conv.1.num_batches_tracked', 'features.6.conv.3.weight', 'features.6.conv.4.weight', 'features.6.conv.4.bias', 'features.6.conv.4.running_mean',
                        'features.6.conv.4.running_var', 'features.6.conv.4.num_batches_tracked', 'features.6.conv.6.weight', 'features.6.conv.7.weight', 'features.6.conv.7.bias', 'features.6.conv.7.running_mean', 'features.6.conv.7.running_var',
                        'features.6.conv.7.num_batches_tracked', 'features.7.conv.0.weight', 'features.7.conv.1.weight', 'features.7.conv.1.bias', 'features.7.conv.1.running_mean', 'features.7.conv.1.running_var', 'features.7.conv.1.num_batches_tracked',
                        'features.7.conv.3.weight', 'features.7.conv.4.weight', 'features.7.conv.4.bias', 'features.7.conv.4.running_mean', 'features.7.conv.4.running_var', 'features.7.conv.4.num_batches_tracked', 'features.7.conv.6.weight',
                        'features.7.conv.7.weight', 'features.7.conv.7.bias', 'features.7.conv.7.running_mean', 'features.7.conv.7.running_var', 'features.7.conv.7.num_batches_tracked', 'features.8.conv.0.weight', 'features.8.conv.1.weight',
                        'features.8.conv.1.bias', 'features.8.conv.1.running_mean', 'features.8.conv.1.running_var', 'features.8.conv.1.num_batches_tracked', 'features.8.conv.3.weight', 'features.8.conv.4.weight', 'features.8.conv.4.bias',
                        'features.8.conv.4.running_mean', 'features.8.conv.4.running_var', 'features.8.conv.4.num_batches_tracked', 'features.8.conv.6.weight', 'features.8.conv.7.weight', 'features.8.conv.7.bias', 'features.8.conv.7.running_mean',
                        'features.8.conv.7.running_var', 'features.8.conv.7.num_batches_tracked', 'features.9.conv.0.weight', 'features.9.conv.1.weight', 'features.9.conv.1.bias', 'features.9.conv.1.running_mean', 'features.9.conv.1.running_var',
                        'features.9.conv.1.num_batches_tracked', 'features.9.conv.3.weight', 'features.9.conv.4.weight', 'features.9.conv.4.bias', 'features.9.conv.4.running_mean', 'features.9.conv.4.running_var', 'features.9.conv.4.num_batches_tracked',
                        'features.9.conv.6.weight', 'features.9.conv.7.weight', 'features.9.conv.7.bias', 'features.9.conv.7.running_mean', 'features.9.conv.7.running_var', 'features.9.conv.7.num_batches_tracked', 'features.10.conv.0.weight',
                        'features.10.conv.1.weight', 'features.10.conv.1.bias', 'features.10.conv.1.running_mean', 'features.10.conv.1.running_var', 'features.10.conv.1.num_batches_tracked', 'features.10.conv.3.weight', 'features.10.conv.4.weight',
                        'features.10.conv.4.bias', 'features.10.conv.4.running_mean', 'features.10.conv.4.running_var', 'features.10.conv.4.num_batches_tracked', 'features.10.conv.6.weight', 'features.10.conv.7.weight', 'features.10.conv.7.bias',
                        'features.10.conv.7.running_mean', 'features.10.conv.7.running_var', 'features.10.conv.7.num_batches_tracked', 'features.11.conv.0.weight', 'features.11.conv.1.weight', 'features.11.conv.1.bias', 'features.11.conv.1.running_mean',
                        'features.11.conv.1.running_var', 'features.11.conv.1.num_batches_tracked', 'features.11.conv.3.weight', 'features.11.conv.4.weight', 'features.11.conv.4.bias', 'features.11.conv.4.running_mean', 'features.11.conv.4.running_var',
                        'features.11.conv.4.num_batches_tracked', 'features.11.conv.6.weight', 'features.11.conv.7.weight', 'features.11.conv.7.bias', 'features.11.conv.7.running_mean', 'features.11.conv.7.running_var', 'features.11.conv.7.num_batches_tracked',
                        'features.12.conv.0.weight', 'features.12.conv.1.weight', 'features.12.conv.1.bias', 'features.12.conv.1.running_mean', 'features.12.conv.1.running_var', 'features.12.conv.1.num_batches_tracked', 'features.12.conv.3.weight',
                        'features.12.conv.4.weight', 'features.12.conv.4.bias', 'features.12.conv.4.running_mean', 'features.12.conv.4.running_var', 'features.12.conv.4.num_batches_tracked', 'features.12.conv.6.weight', 'features.12.conv.7.weight',
                        'features.12.conv.7.bias', 'features.12.conv.7.running_mean', 'features.12.conv.7.running_var', 'features.12.conv.7.num_batches_tracked', 'features.13.conv.0.weight', 'features.13.conv.1.weight', 'features.13.conv.1.bias',
                        'features.13.conv.1.running_mean', 'features.13.conv.1.running_var', 'features.13.conv.1.num_batches_tracked', 'features.13.conv.3.weight', 'features.13.conv.4.weight', 'features.13.conv.4.bias', 'features.13.conv.4.running_mean',
                        'features.13.conv.4.running_var', 'features.13.conv.4.num_batches_tracked', 'features.13.conv.6.weight', 'features.13.conv.7.weight', 'features.13.conv.7.bias', 'features.13.conv.7.running_mean', 'features.13.conv.7.running_var',
                        'features.13.conv.7.num_batches_tracked', 'features.14.conv.0.weight', 'features.14.conv.1.weight', 'features.14.conv.1.bias', 'features.14.conv.1.running_mean', 'features.14.conv.1.running_var', 'features.14.conv.1.num_batches_tracked',
                        'features.14.conv.3.weight', 'features.14.conv.4.weight', 'features.14.conv.4.bias', 'features.14.conv.4.running_mean', 'features.14.conv.4.running_var', 'features.14.conv.4.num_batches_tracked', 'features.14.conv.6.weight',
                        'features.14.conv.7.weight', 'features.14.conv.7.bias', 'features.14.conv.7.running_mean', 'features.14.conv.7.running_var', 'features.14.conv.7.num_batches_tracked', 'features.15.conv.0.weight', 'features.15.conv.1.weight',
                        'features.15.conv.1.bias', 'features.15.conv.1.running_mean', 'features.15.conv.1.running_var', 'features.15.conv.1.num_batches_tracked', 'features.15.conv.3.weight', 'features.15.conv.4.weight', 'features.15.conv.4.bias',
                        'features.15.conv.4.running_mean', 'features.15.conv.4.running_var', 'features.15.conv.4.num_batches_tracked', 'features.15.conv.6.weight', 'features.15.conv.7.weight', 'features.15.conv.7.bias', 'features.15.conv.7.running_mean',
                        'features.15.conv.7.running_var', 'features.15.conv.7.num_batches_tracked', 'features.16.conv.0.weight', 'features.16.conv.1.weight', 'features.16.conv.1.bias', 'features.16.conv.1.running_mean', 'features.16.conv.1.running_var',
                        'features.16.conv.1.num_batches_tracked', 'features.16.conv.3.weight', 'features.16.conv.4.weight', 'features.16.conv.4.bias', 'features.16.conv.4.running_mean', 'features.16.conv.4.running_var', 'features.16.conv.4.num_batches_tracked',
                        'features.16.conv.6.weight', 'features.16.conv.7.weight', 'features.16.conv.7.bias', 'features.16.conv.7.running_mean', 'features.16.conv.7.running_var', 'features.16.conv.7.num_batches_tracked', 'features.17.conv.0.weight',
                        'features.17.conv.1.weight', 'features.17.conv.1.bias', 'features.17.conv.1.running_mean', 'features.17.conv.1.running_var', 'features.17.conv.1.num_batches_tracked', 'features.17.conv.3.weight', 'features.17.conv.4.weight',
                        'features.17.conv.4.bias', 'features.17.conv.4.running_mean', 'features.17.conv.4.running_var', 'features.17.conv.4.num_batches_tracked', 'features.17.conv.6.weight', 'features.17.conv.7.weight', 'features.17.conv.7.bias',
                        'features.17.conv.7.running_mean', 'features.17.conv.7.running_var', 'features.17.conv.7.num_batches_tracked', 'features.18.0.weight', 'features.18.1.weight', 'features.18.1.bias', 'features.18.1.running_mean',
                        'features.18.1.running_var', 'features.18.1.num_batches_tracked']
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

    def __load_pretrained_weights_mobilenet3d_v2_045(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv.0.weight', 'features.1.conv.1.weight',
                        'features.1.conv.1.bias', 'features.1.conv.1.running_mean', 'features.1.conv.1.running_var', 'features.1.conv.1.num_batches_tracked', 'features.1.conv.3.weight', 'features.1.conv.4.weight', 'features.1.conv.4.bias',
                        'features.1.conv.4.running_mean', 'features.1.conv.4.running_var', 'features.1.conv.4.num_batches_tracked', 'features.2.conv.0.weight', 'features.2.conv.1.weight', 'features.2.conv.1.bias', 'features.2.conv.1.running_mean',
                        'features.2.conv.1.running_var', 'features.2.conv.1.num_batches_tracked', 'features.2.conv.3.weight', 'features.2.conv.4.weight', 'features.2.conv.4.bias', 'features.2.conv.4.running_mean', 'features.2.conv.4.running_var',
                        'features.2.conv.4.num_batches_tracked', 'features.2.conv.6.weight', 'features.2.conv.7.weight', 'features.2.conv.7.bias', 'features.2.conv.7.running_mean', 'features.2.conv.7.running_var', 'features.2.conv.7.num_batches_tracked',
                        'features.3.conv.0.weight', 'features.3.conv.1.weight', 'features.3.conv.1.bias', 'features.3.conv.1.running_mean', 'features.3.conv.1.running_var', 'features.3.conv.1.num_batches_tracked', 'features.3.conv.3.weight',
                        'features.3.conv.4.weight', 'features.3.conv.4.bias', 'features.3.conv.4.running_mean', 'features.3.conv.4.running_var', 'features.3.conv.4.num_batches_tracked', 'features.3.conv.6.weight', 'features.3.conv.7.weight',
                        'features.3.conv.7.bias', 'features.3.conv.7.running_mean', 'features.3.conv.7.running_var', 'features.3.conv.7.num_batches_tracked', 'features.4.conv.0.weight', 'features.4.conv.1.weight', 'features.4.conv.1.bias',
                        'features.4.conv.1.running_mean', 'features.4.conv.1.running_var', 'features.4.conv.1.num_batches_tracked', 'features.4.conv.3.weight', 'features.4.conv.4.weight', 'features.4.conv.4.bias', 'features.4.conv.4.running_mean',
                        'features.4.conv.4.running_var', 'features.4.conv.4.num_batches_tracked', 'features.4.conv.6.weight', 'features.4.conv.7.weight', 'features.4.conv.7.bias', 'features.4.conv.7.running_mean', 'features.4.conv.7.running_var',
                        'features.4.conv.7.num_batches_tracked', 'features.5.conv.0.weight', 'features.5.conv.1.weight', 'features.5.conv.1.bias', 'features.5.conv.1.running_mean', 'features.5.conv.1.running_var', 'features.5.conv.1.num_batches_tracked',
                        'features.5.conv.3.weight', 'features.5.conv.4.weight', 'features.5.conv.4.bias', 'features.5.conv.4.running_mean', 'features.5.conv.4.running_var', 'features.5.conv.4.num_batches_tracked', 'features.5.conv.6.weight',
                        'features.5.conv.7.weight', 'features.5.conv.7.bias', 'features.5.conv.7.running_mean', 'features.5.conv.7.running_var', 'features.5.conv.7.num_batches_tracked', 'features.6.conv.0.weight', 'features.6.conv.1.weight',
                        'features.6.conv.1.bias', 'features.6.conv.1.running_mean', 'features.6.conv.1.running_var', 'features.6.conv.1.num_batches_tracked', 'features.6.conv.3.weight', 'features.6.conv.4.weight', 'features.6.conv.4.bias',
                        'features.6.conv.4.running_mean', 'features.6.conv.4.running_var', 'features.6.conv.4.num_batches_tracked', 'features.6.conv.6.weight', 'features.6.conv.7.weight', 'features.6.conv.7.bias', 'features.6.conv.7.running_mean',
                        'features.6.conv.7.running_var', 'features.6.conv.7.num_batches_tracked', 'features.7.conv.0.weight', 'features.7.conv.1.weight', 'features.7.conv.1.bias', 'features.7.conv.1.running_mean', 'features.7.conv.1.running_var',
                        'features.7.conv.1.num_batches_tracked', 'features.7.conv.3.weight', 'features.7.conv.4.weight', 'features.7.conv.4.bias', 'features.7.conv.4.running_mean', 'features.7.conv.4.running_var', 'features.7.conv.4.num_batches_tracked',
                        'features.7.conv.6.weight', 'features.7.conv.7.weight', 'features.7.conv.7.bias', 'features.7.conv.7.running_mean', 'features.7.conv.7.running_var', 'features.7.conv.7.num_batches_tracked', 'features.8.conv.0.weight',
                        'features.8.conv.1.weight', 'features.8.conv.1.bias', 'features.8.conv.1.running_mean', 'features.8.conv.1.running_var', 'features.8.conv.1.num_batches_tracked', 'features.8.conv.3.weight', 'features.8.conv.4.weight',
                        'features.8.conv.4.bias', 'features.8.conv.4.running_mean', 'features.8.conv.4.running_var', 'features.8.conv.4.num_batches_tracked', 'features.8.conv.6.weight', 'features.8.conv.7.weight', 'features.8.conv.7.bias',
                        'features.8.conv.7.running_mean', 'features.8.conv.7.running_var', 'features.8.conv.7.num_batches_tracked', 'features.9.conv.0.weight', 'features.9.conv.1.weight', 'features.9.conv.1.bias', 'features.9.conv.1.running_mean',
                        'features.9.conv.1.running_var', 'features.9.conv.1.num_batches_tracked', 'features.9.conv.3.weight', 'features.9.conv.4.weight', 'features.9.conv.4.bias', 'features.9.conv.4.running_mean', 'features.9.conv.4.running_var',
                        'features.9.conv.4.num_batches_tracked', 'features.9.conv.6.weight', 'features.9.conv.7.weight', 'features.9.conv.7.bias', 'features.9.conv.7.running_mean', 'features.9.conv.7.running_var', 'features.9.conv.7.num_batches_tracked',
                        'features.10.conv.0.weight', 'features.10.conv.1.weight', 'features.10.conv.1.bias', 'features.10.conv.1.running_mean', 'features.10.conv.1.running_var', 'features.10.conv.1.num_batches_tracked', 'features.10.conv.3.weight',
                        'features.10.conv.4.weight', 'features.10.conv.4.bias', 'features.10.conv.4.running_mean', 'features.10.conv.4.running_var', 'features.10.conv.4.num_batches_tracked', 'features.10.conv.6.weight', 'features.10.conv.7.weight',
                        'features.10.conv.7.bias', 'features.10.conv.7.running_mean', 'features.10.conv.7.running_var', 'features.10.conv.7.num_batches_tracked', 'features.11.conv.0.weight', 'features.11.conv.1.weight', 'features.11.conv.1.bias',
                        'features.11.conv.1.running_mean', 'features.11.conv.1.running_var', 'features.11.conv.1.num_batches_tracked', 'features.11.conv.3.weight', 'features.11.conv.4.weight', 'features.11.conv.4.bias', 'features.11.conv.4.running_mean',
                        'features.11.conv.4.running_var', 'features.11.conv.4.num_batches_tracked', 'features.11.conv.6.weight', 'features.11.conv.7.weight', 'features.11.conv.7.bias', 'features.11.conv.7.running_mean', 'features.11.conv.7.running_var',
                        'features.11.conv.7.num_batches_tracked', 'features.12.conv.0.weight', 'features.12.conv.1.weight', 'features.12.conv.1.bias', 'features.12.conv.1.running_mean', 'features.12.conv.1.running_var', 'features.12.conv.1.num_batches_tracked',
                        'features.12.conv.3.weight', 'features.12.conv.4.weight', 'features.12.conv.4.bias', 'features.12.conv.4.running_mean', 'features.12.conv.4.running_var', 'features.12.conv.4.num_batches_tracked', 'features.12.conv.6.weight',
                        'features.12.conv.7.weight', 'features.12.conv.7.bias', 'features.12.conv.7.running_mean', 'features.12.conv.7.running_var', 'features.12.conv.7.num_batches_tracked', 'features.13.conv.0.weight', 'features.13.conv.1.weight',
                        'features.13.conv.1.bias', 'features.13.conv.1.running_mean', 'features.13.conv.1.running_var', 'features.13.conv.1.num_batches_tracked', 'features.13.conv.3.weight', 'features.13.conv.4.weight', 'features.13.conv.4.bias',
                        'features.13.conv.4.running_mean', 'features.13.conv.4.running_var', 'features.13.conv.4.num_batches_tracked', 'features.13.conv.6.weight', 'features.13.conv.7.weight', 'features.13.conv.7.bias', 'features.13.conv.7.running_mean',
                        'features.13.conv.7.running_var', 'features.13.conv.7.num_batches_tracked', 'features.14.conv.0.weight', 'features.14.conv.1.weight', 'features.14.conv.1.bias', 'features.14.conv.1.running_mean', 'features.14.conv.1.running_var',
                        'features.14.conv.1.num_batches_tracked', 'features.14.conv.3.weight', 'features.14.conv.4.weight', 'features.14.conv.4.bias', 'features.14.conv.4.running_mean', 'features.14.conv.4.running_var', 'features.14.conv.4.num_batches_tracked',
                        'features.14.conv.6.weight', 'features.14.conv.7.weight', 'features.14.conv.7.bias', 'features.14.conv.7.running_mean', 'features.14.conv.7.running_var', 'features.14.conv.7.num_batches_tracked', 'features.15.conv.0.weight',
                        'features.15.conv.1.weight', 'features.15.conv.1.bias', 'features.15.conv.1.running_mean', 'features.15.conv.1.running_var', 'features.15.conv.1.num_batches_tracked', 'features.15.conv.3.weight', 'features.15.conv.4.weight',
                        'features.15.conv.4.bias', 'features.15.conv.4.running_mean', 'features.15.conv.4.running_var', 'features.15.conv.4.num_batches_tracked', 'features.15.conv.6.weight', 'features.15.conv.7.weight', 'features.15.conv.7.bias',
                        'features.15.conv.7.running_mean', 'features.15.conv.7.running_var', 'features.15.conv.7.num_batches_tracked', 'features.16.conv.0.weight', 'features.16.conv.1.weight', 'features.16.conv.1.bias', 'features.16.conv.1.running_mean',
                        'features.16.conv.1.running_var', 'features.16.conv.1.num_batches_tracked', 'features.16.conv.3.weight', 'features.16.conv.4.weight', 'features.16.conv.4.bias', 'features.16.conv.4.running_mean', 'features.16.conv.4.running_var',
                        'features.16.conv.4.num_batches_tracked', 'features.16.conv.6.weight', 'features.16.conv.7.weight', 'features.16.conv.7.bias', 'features.16.conv.7.running_mean', 'features.16.conv.7.running_var', 'features.16.conv.7.num_batches_tracked',
                        'features.17.conv.0.weight', 'features.17.conv.1.weight', 'features.17.conv.1.bias', 'features.17.conv.1.running_mean', 'features.17.conv.1.running_var', 'features.17.conv.1.num_batches_tracked', 'features.17.conv.3.weight',
                        'features.17.conv.4.weight', 'features.17.conv.4.bias', 'features.17.conv.4.running_mean', 'features.17.conv.4.running_var', 'features.17.conv.4.num_batches_tracked', 'features.17.conv.6.weight', 'features.17.conv.7.weight',
                        'features.17.conv.7.bias', 'features.17.conv.7.running_mean', 'features.17.conv.7.running_var', 'features.17.conv.7.num_batches_tracked', 'features.18.0.weight', 'features.18.1.weight', 'features.18.1.bias', 'features.18.1.running_mean',
                        'features.18.1.running_var', 'features.18.1.num_batches_tracked']
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

    def __load_pretrained_weights_mobilenet3d_v2_10(self):
        corresp_name = ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.conv.0.weight', 'features.1.conv.1.weight',
                        'features.1.conv.1.bias', 'features.1.conv.1.running_mean', 'features.1.conv.1.running_var', 'features.1.conv.1.num_batches_tracked', 'features.1.conv.3.weight', 'features.1.conv.4.weight', 'features.1.conv.4.bias',
                        'features.1.conv.4.running_mean', 'features.1.conv.4.running_var', 'features.1.conv.4.num_batches_tracked', 'features.2.conv.0.weight', 'features.2.conv.1.weight', 'features.2.conv.1.bias', 'features.2.conv.1.running_mean',
                        'features.2.conv.1.running_var', 'features.2.conv.1.num_batches_tracked', 'features.2.conv.3.weight', 'features.2.conv.4.weight', 'features.2.conv.4.bias', 'features.2.conv.4.running_mean', 'features.2.conv.4.running_var',
                        'features.2.conv.4.num_batches_tracked', 'features.2.conv.6.weight', 'features.2.conv.7.weight', 'features.2.conv.7.bias', 'features.2.conv.7.running_mean', 'features.2.conv.7.running_var', 'features.2.conv.7.num_batches_tracked',
                        'features.3.conv.0.weight', 'features.3.conv.1.weight', 'features.3.conv.1.bias', 'features.3.conv.1.running_mean', 'features.3.conv.1.running_var', 'features.3.conv.1.num_batches_tracked', 'features.3.conv.3.weight',
                        'features.3.conv.4.weight', 'features.3.conv.4.bias', 'features.3.conv.4.running_mean', 'features.3.conv.4.running_var', 'features.3.conv.4.num_batches_tracked', 'features.3.conv.6.weight', 'features.3.conv.7.weight',
                        'features.3.conv.7.bias', 'features.3.conv.7.running_mean', 'features.3.conv.7.running_var', 'features.3.conv.7.num_batches_tracked', 'features.4.conv.0.weight', 'features.4.conv.1.weight', 'features.4.conv.1.bias',
                        'features.4.conv.1.running_mean', 'features.4.conv.1.running_var', 'features.4.conv.1.num_batches_tracked', 'features.4.conv.3.weight', 'features.4.conv.4.weight', 'features.4.conv.4.bias', 'features.4.conv.4.running_mean',
                        'features.4.conv.4.running_var', 'features.4.conv.4.num_batches_tracked', 'features.4.conv.6.weight', 'features.4.conv.7.weight', 'features.4.conv.7.bias', 'features.4.conv.7.running_mean', 'features.4.conv.7.running_var',
                        'features.4.conv.7.num_batches_tracked', 'features.5.conv.0.weight', 'features.5.conv.1.weight', 'features.5.conv.1.bias', 'features.5.conv.1.running_mean', 'features.5.conv.1.running_var', 'features.5.conv.1.num_batches_tracked',
                        'features.5.conv.3.weight', 'features.5.conv.4.weight', 'features.5.conv.4.bias', 'features.5.conv.4.running_mean', 'features.5.conv.4.running_var', 'features.5.conv.4.num_batches_tracked', 'features.5.conv.6.weight',
                        'features.5.conv.7.weight', 'features.5.conv.7.bias', 'features.5.conv.7.running_mean', 'features.5.conv.7.running_var', 'features.5.conv.7.num_batches_tracked', 'features.6.conv.0.weight', 'features.6.conv.1.weight',
                        'features.6.conv.1.bias', 'features.6.conv.1.running_mean', 'features.6.conv.1.running_var', 'features.6.conv.1.num_batches_tracked', 'features.6.conv.3.weight', 'features.6.conv.4.weight', 'features.6.conv.4.bias',
                        'features.6.conv.4.running_mean', 'features.6.conv.4.running_var', 'features.6.conv.4.num_batches_tracked', 'features.6.conv.6.weight', 'features.6.conv.7.weight', 'features.6.conv.7.bias', 'features.6.conv.7.running_mean',
                        'features.6.conv.7.running_var', 'features.6.conv.7.num_batches_tracked', 'features.7.conv.0.weight', 'features.7.conv.1.weight', 'features.7.conv.1.bias', 'features.7.conv.1.running_mean', 'features.7.conv.1.running_var',
                        'features.7.conv.1.num_batches_tracked', 'features.7.conv.3.weight', 'features.7.conv.4.weight', 'features.7.conv.4.bias', 'features.7.conv.4.running_mean', 'features.7.conv.4.running_var', 'features.7.conv.4.num_batches_tracked',
                        'features.7.conv.6.weight', 'features.7.conv.7.weight', 'features.7.conv.7.bias', 'features.7.conv.7.running_mean', 'features.7.conv.7.running_var', 'features.7.conv.7.num_batches_tracked', 'features.8.conv.0.weight',
                        'features.8.conv.1.weight', 'features.8.conv.1.bias', 'features.8.conv.1.running_mean', 'features.8.conv.1.running_var', 'features.8.conv.1.num_batches_tracked', 'features.8.conv.3.weight', 'features.8.conv.4.weight',
                        'features.8.conv.4.bias', 'features.8.conv.4.running_mean', 'features.8.conv.4.running_var', 'features.8.conv.4.num_batches_tracked', 'features.8.conv.6.weight', 'features.8.conv.7.weight', 'features.8.conv.7.bias',
                        'features.8.conv.7.running_mean', 'features.8.conv.7.running_var', 'features.8.conv.7.num_batches_tracked', 'features.9.conv.0.weight', 'features.9.conv.1.weight', 'features.9.conv.1.bias', 'features.9.conv.1.running_mean',
                        'features.9.conv.1.running_var', 'features.9.conv.1.num_batches_tracked', 'features.9.conv.3.weight', 'features.9.conv.4.weight', 'features.9.conv.4.bias', 'features.9.conv.4.running_mean', 'features.9.conv.4.running_var',
                        'features.9.conv.4.num_batches_tracked', 'features.9.conv.6.weight', 'features.9.conv.7.weight', 'features.9.conv.7.bias', 'features.9.conv.7.running_mean', 'features.9.conv.7.running_var', 'features.9.conv.7.num_batches_tracked',
                        'features.10.conv.0.weight', 'features.10.conv.1.weight', 'features.10.conv.1.bias', 'features.10.conv.1.running_mean', 'features.10.conv.1.running_var', 'features.10.conv.1.num_batches_tracked', 'features.10.conv.3.weight',
                        'features.10.conv.4.weight', 'features.10.conv.4.bias', 'features.10.conv.4.running_mean', 'features.10.conv.4.running_var', 'features.10.conv.4.num_batches_tracked', 'features.10.conv.6.weight', 'features.10.conv.7.weight',
                        'features.10.conv.7.bias', 'features.10.conv.7.running_mean', 'features.10.conv.7.running_var', 'features.10.conv.7.num_batches_tracked', 'features.11.conv.0.weight', 'features.11.conv.1.weight', 'features.11.conv.1.bias',
                        'features.11.conv.1.running_mean', 'features.11.conv.1.running_var', 'features.11.conv.1.num_batches_tracked', 'features.11.conv.3.weight', 'features.11.conv.4.weight', 'features.11.conv.4.bias', 'features.11.conv.4.running_mean',
                        'features.11.conv.4.running_var', 'features.11.conv.4.num_batches_tracked', 'features.11.conv.6.weight', 'features.11.conv.7.weight', 'features.11.conv.7.bias', 'features.11.conv.7.running_mean', 'features.11.conv.7.running_var',
                        'features.11.conv.7.num_batches_tracked', 'features.12.conv.0.weight', 'features.12.conv.1.weight', 'features.12.conv.1.bias', 'features.12.conv.1.running_mean', 'features.12.conv.1.running_var', 'features.12.conv.1.num_batches_tracked',
                        'features.12.conv.3.weight', 'features.12.conv.4.weight', 'features.12.conv.4.bias', 'features.12.conv.4.running_mean', 'features.12.conv.4.running_var', 'features.12.conv.4.num_batches_tracked', 'features.12.conv.6.weight',
                        'features.12.conv.7.weight', 'features.12.conv.7.bias', 'features.12.conv.7.running_mean', 'features.12.conv.7.running_var', 'features.12.conv.7.num_batches_tracked', 'features.13.conv.0.weight', 'features.13.conv.1.weight',
                        'features.13.conv.1.bias', 'features.13.conv.1.running_mean', 'features.13.conv.1.running_var', 'features.13.conv.1.num_batches_tracked', 'features.13.conv.3.weight', 'features.13.conv.4.weight', 'features.13.conv.4.bias',
                        'features.13.conv.4.running_mean', 'features.13.conv.4.running_var', 'features.13.conv.4.num_batches_tracked', 'features.13.conv.6.weight', 'features.13.conv.7.weight', 'features.13.conv.7.bias', 'features.13.conv.7.running_mean',
                        'features.13.conv.7.running_var', 'features.13.conv.7.num_batches_tracked', 'features.14.conv.0.weight', 'features.14.conv.1.weight', 'features.14.conv.1.bias', 'features.14.conv.1.running_mean', 'features.14.conv.1.running_var',
                        'features.14.conv.1.num_batches_tracked', 'features.14.conv.3.weight', 'features.14.conv.4.weight', 'features.14.conv.4.bias', 'features.14.conv.4.running_mean', 'features.14.conv.4.running_var', 'features.14.conv.4.num_batches_tracked',
                        'features.14.conv.6.weight', 'features.14.conv.7.weight', 'features.14.conv.7.bias', 'features.14.conv.7.running_mean', 'features.14.conv.7.running_var', 'features.14.conv.7.num_batches_tracked', 'features.15.conv.0.weight',
                        'features.15.conv.1.weight', 'features.15.conv.1.bias', 'features.15.conv.1.running_mean', 'features.15.conv.1.running_var', 'features.15.conv.1.num_batches_tracked', 'features.15.conv.3.weight', 'features.15.conv.4.weight',
                        'features.15.conv.4.bias', 'features.15.conv.4.running_mean', 'features.15.conv.4.running_var', 'features.15.conv.4.num_batches_tracked', 'features.15.conv.6.weight', 'features.15.conv.7.weight', 'features.15.conv.7.bias',
                        'features.15.conv.7.running_mean', 'features.15.conv.7.running_var', 'features.15.conv.7.num_batches_tracked', 'features.16.conv.0.weight', 'features.16.conv.1.weight', 'features.16.conv.1.bias', 'features.16.conv.1.running_mean',
                        'features.16.conv.1.running_var', 'features.16.conv.1.num_batches_tracked', 'features.16.conv.3.weight', 'features.16.conv.4.weight', 'features.16.conv.4.bias', 'features.16.conv.4.running_mean', 'features.16.conv.4.running_var',
                        'features.16.conv.4.num_batches_tracked', 'features.16.conv.6.weight', 'features.16.conv.7.weight', 'features.16.conv.7.bias', 'features.16.conv.7.running_mean', 'features.16.conv.7.running_var', 'features.16.conv.7.num_batches_tracked',
                        'features.17.conv.0.weight', 'features.17.conv.1.weight', 'features.17.conv.1.bias', 'features.17.conv.1.running_mean', 'features.17.conv.1.running_var', 'features.17.conv.1.num_batches_tracked', 'features.17.conv.3.weight',
                        'features.17.conv.4.weight', 'features.17.conv.4.bias', 'features.17.conv.4.running_mean', 'features.17.conv.4.running_var', 'features.17.conv.4.num_batches_tracked', 'features.17.conv.6.weight', 'features.17.conv.7.weight',
                        'features.17.conv.7.bias', 'features.17.conv.7.running_mean', 'features.17.conv.7.running_var', 'features.17.conv.7.num_batches_tracked', 'features.18.0.weight', 'features.18.1.weight', 'features.18.1.bias', 'features.18.1.running_mean',
                        'features.18.1.running_var', 'features.18.1.num_batches_tracked']
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = MobileNet3D_v2(num_classes=101, width_mult=1.0, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_1.0x_RGB_16_best.pth')
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)