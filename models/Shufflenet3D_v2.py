# https://github.com/okankop/Efficient-3DCNNs/blob/master/models/shufflenetv2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False), nn.BatchNorm3d(oup), nn.ReLU(inplace=True))

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv3d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup), nn.ReLU(inplace=True))

def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, depth, height, width)
    # permute
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        oup_inc = oup // 2
        if self.stride == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False), # pw
                                        nn.BatchNorm3d(oup_inc),
                                        nn.ReLU(inplace=True),
                                        # dw
                                        nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                                        nn.BatchNorm3d(oup_inc),
                                        # pw-linear
                                        nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                                        nn.BatchNorm3d(oup_inc),
                                        nn.ReLU(inplace=True))
        else:
            self.banch1 = nn.Sequential(nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False), # dw
                                        nn.BatchNorm3d(inp),
                                        # pw-linear
                                        nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                                        nn.BatchNorm3d(oup_inc),
                                        nn.ReLU(inplace=True))
            self.banch2 = nn.Sequential(nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False), # pw
                                        nn.BatchNorm3d(oup_inc),
                                        nn.ReLU(inplace=True),
                                        # dw
                                        nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                                        nn.BatchNorm3d(oup_inc),
                                        # pw-linear
                                        nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                                        nn.BatchNorm3d(oup_inc),
                                        nn.ReLU(inplace=True))

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))
        return channel_shuffle(out, 2)

class ShuffleNet3D_v2(nn.Module):
    def __init__(self, num_classes=600, width_mult=1., pretrained=None):
        super(ShuffleNet3D_v2, self).__init__()
        self.pretrained = pretrained
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24, 32, 64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError("""{} groups is not supported for 1x1 Grouped Convolutions""".format(width_mult))
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=(1, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        # building last several layers
        self.conv_last = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
        # building classifier
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.stage_out_channels[-1], num_classes))
        if pretrained:
            if width_mult == 0.25:
                print('Loading pretrained weights for shufflenet3D_v2 with width = 0.25...')
                self.__load_pretrained_weights_shufflenet3d_v2_025()
            elif width_mult == 1.0:
                print('Loading pretrained weights for shufflenet3D_v2 with width = 1.0...')
                self.__load_pretrained_weights_shufflenet3d_v2_10()
            elif width_mult == 1.5:
                print('Loading pretrained weights for shufflenet3D_v2 with width = 1.5...')
                self.__load_pretrained_weights_shufflenet3d_v2_15()
            elif width_mult == 2.0:
                print('Loading pretrained weights for shufflenet3D_v2 with width = 2.0...')
                self.__load_pretrained_weights_shufflenet3d_v2_20()

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def __load_pretrained_weights_shufflenet3d_v2_025(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'features.0.banch1.0.weight', 'features.0.banch1.1.weight', 'features.0.banch1.1.bias',
                        'features.0.banch1.1.running_mean', 'features.0.banch1.1.running_var', 'features.0.banch1.1.num_batches_tracked', 'features.0.banch1.2.weight', 'features.0.banch1.3.weight', 'features.0.banch1.3.bias', 'features.0.banch1.3.running_mean',
                        'features.0.banch1.3.running_var', 'features.0.banch1.3.num_batches_tracked', 'features.0.banch2.0.weight', 'features.0.banch2.1.weight', 'features.0.banch2.1.bias', 'features.0.banch2.1.running_mean', 'features.0.banch2.1.running_var',
                        'features.0.banch2.1.num_batches_tracked', 'features.0.banch2.3.weight', 'features.0.banch2.4.weight', 'features.0.banch2.4.bias', 'features.0.banch2.4.running_mean', 'features.0.banch2.4.running_var',
                        'features.0.banch2.4.num_batches_tracked', 'features.0.banch2.5.weight', 'features.0.banch2.6.weight', 'features.0.banch2.6.bias', 'features.0.banch2.6.running_mean', 'features.0.banch2.6.running_var',
                        'features.0.banch2.6.num_batches_tracked', 'features.1.banch2.0.weight', 'features.1.banch2.1.weight', 'features.1.banch2.1.bias', 'features.1.banch2.1.running_mean', 'features.1.banch2.1.running_var',
                        'features.1.banch2.1.num_batches_tracked', 'features.1.banch2.3.weight', 'features.1.banch2.4.weight', 'features.1.banch2.4.bias', 'features.1.banch2.4.running_mean', 'features.1.banch2.4.running_var',
                        'features.1.banch2.4.num_batches_tracked', 'features.1.banch2.5.weight', 'features.1.banch2.6.weight', 'features.1.banch2.6.bias', 'features.1.banch2.6.running_mean', 'features.1.banch2.6.running_var',
                        'features.1.banch2.6.num_batches_tracked', 'features.2.banch2.0.weight', 'features.2.banch2.1.weight', 'features.2.banch2.1.bias', 'features.2.banch2.1.running_mean', 'features.2.banch2.1.running_var',
                        'features.2.banch2.1.num_batches_tracked', 'features.2.banch2.3.weight', 'features.2.banch2.4.weight', 'features.2.banch2.4.bias', 'features.2.banch2.4.running_mean', 'features.2.banch2.4.running_var',
                        'features.2.banch2.4.num_batches_tracked', 'features.2.banch2.5.weight', 'features.2.banch2.6.weight', 'features.2.banch2.6.bias', 'features.2.banch2.6.running_mean', 'features.2.banch2.6.running_var',
                        'features.2.banch2.6.num_batches_tracked', 'features.3.banch2.0.weight', 'features.3.banch2.1.weight', 'features.3.banch2.1.bias', 'features.3.banch2.1.running_mean', 'features.3.banch2.1.running_var',
                        'features.3.banch2.1.num_batches_tracked', 'features.3.banch2.3.weight', 'features.3.banch2.4.weight', 'features.3.banch2.4.bias', 'features.3.banch2.4.running_mean', 'features.3.banch2.4.running_var',
                        'features.3.banch2.4.num_batches_tracked', 'features.3.banch2.5.weight', 'features.3.banch2.6.weight', 'features.3.banch2.6.bias', 'features.3.banch2.6.running_mean', 'features.3.banch2.6.running_var',
                        'features.3.banch2.6.num_batches_tracked', 'features.4.banch1.0.weight', 'features.4.banch1.1.weight', 'features.4.banch1.1.bias', 'features.4.banch1.1.running_mean', 'features.4.banch1.1.running_var',
                        'features.4.banch1.1.num_batches_tracked', 'features.4.banch1.2.weight', 'features.4.banch1.3.weight', 'features.4.banch1.3.bias', 'features.4.banch1.3.running_mean', 'features.4.banch1.3.running_var',
                        'features.4.banch1.3.num_batches_tracked', 'features.4.banch2.0.weight', 'features.4.banch2.1.weight', 'features.4.banch2.1.bias', 'features.4.banch2.1.running_mean', 'features.4.banch2.1.running_var',
                        'features.4.banch2.1.num_batches_tracked', 'features.4.banch2.3.weight', 'features.4.banch2.4.weight', 'features.4.banch2.4.bias', 'features.4.banch2.4.running_mean', 'features.4.banch2.4.running_var',
                        'features.4.banch2.4.num_batches_tracked', 'features.4.banch2.5.weight', 'features.4.banch2.6.weight', 'features.4.banch2.6.bias', 'features.4.banch2.6.running_mean', 'features.4.banch2.6.running_var',
                        'features.4.banch2.6.num_batches_tracked', 'features.5.banch2.0.weight', 'features.5.banch2.1.weight', 'features.5.banch2.1.bias', 'features.5.banch2.1.running_mean', 'features.5.banch2.1.running_var',
                        'features.5.banch2.1.num_batches_tracked', 'features.5.banch2.3.weight', 'features.5.banch2.4.weight', 'features.5.banch2.4.bias', 'features.5.banch2.4.running_mean', 'features.5.banch2.4.running_var',
                        'features.5.banch2.4.num_batches_tracked', 'features.5.banch2.5.weight', 'features.5.banch2.6.weight', 'features.5.banch2.6.bias', 'features.5.banch2.6.running_mean', 'features.5.banch2.6.running_var',
                        'features.5.banch2.6.num_batches_tracked', 'features.6.banch2.0.weight', 'features.6.banch2.1.weight', 'features.6.banch2.1.bias', 'features.6.banch2.1.running_mean', 'features.6.banch2.1.running_var',
                        'features.6.banch2.1.num_batches_tracked', 'features.6.banch2.3.weight', 'features.6.banch2.4.weight', 'features.6.banch2.4.bias', 'features.6.banch2.4.running_mean', 'features.6.banch2.4.running_var',
                        'features.6.banch2.4.num_batches_tracked', 'features.6.banch2.5.weight', 'features.6.banch2.6.weight', 'features.6.banch2.6.bias', 'features.6.banch2.6.running_mean', 'features.6.banch2.6.running_var',
                        'features.6.banch2.6.num_batches_tracked', 'features.7.banch2.0.weight', 'features.7.banch2.1.weight', 'features.7.banch2.1.bias', 'features.7.banch2.1.running_mean', 'features.7.banch2.1.running_var',
                        'features.7.banch2.1.num_batches_tracked', 'features.7.banch2.3.weight', 'features.7.banch2.4.weight', 'features.7.banch2.4.bias', 'features.7.banch2.4.running_mean', 'features.7.banch2.4.running_var',
                        'features.7.banch2.4.num_batches_tracked', 'features.7.banch2.5.weight', 'features.7.banch2.6.weight', 'features.7.banch2.6.bias', 'features.7.banch2.6.running_mean', 'features.7.banch2.6.running_var',
                        'features.7.banch2.6.num_batches_tracked', 'features.8.banch2.0.weight', 'features.8.banch2.1.weight', 'features.8.banch2.1.bias', 'features.8.banch2.1.running_mean', 'features.8.banch2.1.running_var',
                        'features.8.banch2.1.num_batches_tracked', 'features.8.banch2.3.weight', 'features.8.banch2.4.weight', 'features.8.banch2.4.bias', 'features.8.banch2.4.running_mean', 'features.8.banch2.4.running_var',
                        'features.8.banch2.4.num_batches_tracked', 'features.8.banch2.5.weight', 'features.8.banch2.6.weight', 'features.8.banch2.6.bias', 'features.8.banch2.6.running_mean', 'features.8.banch2.6.running_var',
                        'features.8.banch2.6.num_batches_tracked', 'features.9.banch2.0.weight', 'features.9.banch2.1.weight', 'features.9.banch2.1.bias', 'features.9.banch2.1.running_mean', 'features.9.banch2.1.running_var',
                        'features.9.banch2.1.num_batches_tracked', 'features.9.banch2.3.weight', 'features.9.banch2.4.weight', 'features.9.banch2.4.bias', 'features.9.banch2.4.running_mean', 'features.9.banch2.4.running_var',
                        'features.9.banch2.4.num_batches_tracked', 'features.9.banch2.5.weight', 'features.9.banch2.6.weight', 'features.9.banch2.6.bias', 'features.9.banch2.6.running_mean', 'features.9.banch2.6.running_var',
                        'features.9.banch2.6.num_batches_tracked', 'features.10.banch2.0.weight', 'features.10.banch2.1.weight', 'features.10.banch2.1.bias', 'features.10.banch2.1.running_mean', 'features.10.banch2.1.running_var',
                        'features.10.banch2.1.num_batches_tracked', 'features.10.banch2.3.weight', 'features.10.banch2.4.weight', 'features.10.banch2.4.bias', 'features.10.banch2.4.running_mean', 'features.10.banch2.4.running_var',
                        'features.10.banch2.4.num_batches_tracked', 'features.10.banch2.5.weight', 'features.10.banch2.6.weight', 'features.10.banch2.6.bias', 'features.10.banch2.6.running_mean', 'features.10.banch2.6.running_var',
                        'features.10.banch2.6.num_batches_tracked', 'features.11.banch2.0.weight', 'features.11.banch2.1.weight', 'features.11.banch2.1.bias', 'features.11.banch2.1.running_mean', 'features.11.banch2.1.running_var',
                        'features.11.banch2.1.num_batches_tracked', 'features.11.banch2.3.weight', 'features.11.banch2.4.weight', 'features.11.banch2.4.bias', 'features.11.banch2.4.running_mean', 'features.11.banch2.4.running_var',
                        'features.11.banch2.4.num_batches_tracked', 'features.11.banch2.5.weight', 'features.11.banch2.6.weight', 'features.11.banch2.6.bias', 'features.11.banch2.6.running_mean', 'features.11.banch2.6.running_var',
                        'features.11.banch2.6.num_batches_tracked', 'features.12.banch1.0.weight', 'features.12.banch1.1.weight', 'features.12.banch1.1.bias', 'features.12.banch1.1.running_mean', 'features.12.banch1.1.running_var',
                        'features.12.banch1.1.num_batches_tracked', 'features.12.banch1.2.weight', 'features.12.banch1.3.weight', 'features.12.banch1.3.bias', 'features.12.banch1.3.running_mean', 'features.12.banch1.3.running_var',
                        'features.12.banch1.3.num_batches_tracked', 'features.12.banch2.0.weight', 'features.12.banch2.1.weight', 'features.12.banch2.1.bias', 'features.12.banch2.1.running_mean', 'features.12.banch2.1.running_var',
                        'features.12.banch2.1.num_batches_tracked', 'features.12.banch2.3.weight', 'features.12.banch2.4.weight', 'features.12.banch2.4.bias', 'features.12.banch2.4.running_mean', 'features.12.banch2.4.running_var',
                        'features.12.banch2.4.num_batches_tracked', 'features.12.banch2.5.weight', 'features.12.banch2.6.weight', 'features.12.banch2.6.bias', 'features.12.banch2.6.running_mean', 'features.12.banch2.6.running_var',
                        'features.12.banch2.6.num_batches_tracked', 'features.13.banch2.0.weight', 'features.13.banch2.1.weight', 'features.13.banch2.1.bias', 'features.13.banch2.1.running_mean', 'features.13.banch2.1.running_var',
                        'features.13.banch2.1.num_batches_tracked', 'features.13.banch2.3.weight', 'features.13.banch2.4.weight', 'features.13.banch2.4.bias', 'features.13.banch2.4.running_mean', 'features.13.banch2.4.running_var',
                        'features.13.banch2.4.num_batches_tracked', 'features.13.banch2.5.weight', 'features.13.banch2.6.weight', 'features.13.banch2.6.bias', 'features.13.banch2.6.running_mean', 'features.13.banch2.6.running_var',
                        'features.13.banch2.6.num_batches_tracked', 'features.14.banch2.0.weight', 'features.14.banch2.1.weight', 'features.14.banch2.1.bias', 'features.14.banch2.1.running_mean', 'features.14.banch2.1.running_var',
                        'features.14.banch2.1.num_batches_tracked', 'features.14.banch2.3.weight', 'features.14.banch2.4.weight', 'features.14.banch2.4.bias', 'features.14.banch2.4.running_mean', 'features.14.banch2.4.running_var',
                        'features.14.banch2.4.num_batches_tracked', 'features.14.banch2.5.weight', 'features.14.banch2.6.weight', 'features.14.banch2.6.bias', 'features.14.banch2.6.running_mean', 'features.14.banch2.6.running_var',
                        'features.14.banch2.6.num_batches_tracked', 'features.15.banch2.0.weight', 'features.15.banch2.1.weight', 'features.15.banch2.1.bias', 'features.15.banch2.1.running_mean', 'features.15.banch2.1.running_var',
                        'features.15.banch2.1.num_batches_tracked', 'features.15.banch2.3.weight', 'features.15.banch2.4.weight', 'features.15.banch2.4.bias', 'features.15.banch2.4.running_mean', 'features.15.banch2.4.running_var',
                        'features.15.banch2.4.num_batches_tracked', 'features.15.banch2.5.weight', 'features.15.banch2.6.weight', 'features.15.banch2.6.bias', 'features.15.banch2.6.running_mean', 'features.15.banch2.6.running_var',
                        'features.15.banch2.6.num_batches_tracked', 'conv_last.0.weight', 'conv_last.1.weight', 'conv_last.1.bias', 'conv_last.1.running_mean', 'conv_last.1.running_var', 'conv_last.1.num_batches_tracked']
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

    def __load_pretrained_weights_shufflenet3d_v2_10(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'features.0.banch1.0.weight', 'features.0.banch1.1.weight', 'features.0.banch1.1.bias',
                        'features.0.banch1.1.running_mean', 'features.0.banch1.1.running_var', 'features.0.banch1.1.num_batches_tracked', 'features.0.banch1.2.weight', 'features.0.banch1.3.weight', 'features.0.banch1.3.bias', 'features.0.banch1.3.running_mean',
                        'features.0.banch1.3.running_var', 'features.0.banch1.3.num_batches_tracked', 'features.0.banch2.0.weight', 'features.0.banch2.1.weight', 'features.0.banch2.1.bias', 'features.0.banch2.1.running_mean', 'features.0.banch2.1.running_var',
                        'features.0.banch2.1.num_batches_tracked', 'features.0.banch2.3.weight', 'features.0.banch2.4.weight', 'features.0.banch2.4.bias', 'features.0.banch2.4.running_mean', 'features.0.banch2.4.running_var',
                        'features.0.banch2.4.num_batches_tracked', 'features.0.banch2.5.weight', 'features.0.banch2.6.weight', 'features.0.banch2.6.bias', 'features.0.banch2.6.running_mean', 'features.0.banch2.6.running_var',
                        'features.0.banch2.6.num_batches_tracked', 'features.1.banch2.0.weight', 'features.1.banch2.1.weight', 'features.1.banch2.1.bias', 'features.1.banch2.1.running_mean', 'features.1.banch2.1.running_var',
                        'features.1.banch2.1.num_batches_tracked', 'features.1.banch2.3.weight', 'features.1.banch2.4.weight', 'features.1.banch2.4.bias', 'features.1.banch2.4.running_mean', 'features.1.banch2.4.running_var',
                        'features.1.banch2.4.num_batches_tracked', 'features.1.banch2.5.weight', 'features.1.banch2.6.weight', 'features.1.banch2.6.bias', 'features.1.banch2.6.running_mean', 'features.1.banch2.6.running_var',
                        'features.1.banch2.6.num_batches_tracked', 'features.2.banch2.0.weight', 'features.2.banch2.1.weight', 'features.2.banch2.1.bias', 'features.2.banch2.1.running_mean', 'features.2.banch2.1.running_var',
                        'features.2.banch2.1.num_batches_tracked', 'features.2.banch2.3.weight', 'features.2.banch2.4.weight', 'features.2.banch2.4.bias', 'features.2.banch2.4.running_mean', 'features.2.banch2.4.running_var',
                        'features.2.banch2.4.num_batches_tracked', 'features.2.banch2.5.weight', 'features.2.banch2.6.weight', 'features.2.banch2.6.bias', 'features.2.banch2.6.running_mean', 'features.2.banch2.6.running_var',
                        'features.2.banch2.6.num_batches_tracked', 'features.3.banch2.0.weight', 'features.3.banch2.1.weight', 'features.3.banch2.1.bias', 'features.3.banch2.1.running_mean', 'features.3.banch2.1.running_var',
                        'features.3.banch2.1.num_batches_tracked', 'features.3.banch2.3.weight', 'features.3.banch2.4.weight', 'features.3.banch2.4.bias', 'features.3.banch2.4.running_mean', 'features.3.banch2.4.running_var',
                        'features.3.banch2.4.num_batches_tracked', 'features.3.banch2.5.weight', 'features.3.banch2.6.weight', 'features.3.banch2.6.bias', 'features.3.banch2.6.running_mean', 'features.3.banch2.6.running_var',
                        'features.3.banch2.6.num_batches_tracked', 'features.4.banch1.0.weight', 'features.4.banch1.1.weight', 'features.4.banch1.1.bias', 'features.4.banch1.1.running_mean', 'features.4.banch1.1.running_var',
                        'features.4.banch1.1.num_batches_tracked', 'features.4.banch1.2.weight', 'features.4.banch1.3.weight', 'features.4.banch1.3.bias', 'features.4.banch1.3.running_mean', 'features.4.banch1.3.running_var',
                        'features.4.banch1.3.num_batches_tracked', 'features.4.banch2.0.weight', 'features.4.banch2.1.weight', 'features.4.banch2.1.bias', 'features.4.banch2.1.running_mean', 'features.4.banch2.1.running_var',
                        'features.4.banch2.1.num_batches_tracked', 'features.4.banch2.3.weight', 'features.4.banch2.4.weight', 'features.4.banch2.4.bias', 'features.4.banch2.4.running_mean', 'features.4.banch2.4.running_var',
                        'features.4.banch2.4.num_batches_tracked', 'features.4.banch2.5.weight', 'features.4.banch2.6.weight', 'features.4.banch2.6.bias', 'features.4.banch2.6.running_mean', 'features.4.banch2.6.running_var',
                        'features.4.banch2.6.num_batches_tracked', 'features.5.banch2.0.weight', 'features.5.banch2.1.weight', 'features.5.banch2.1.bias', 'features.5.banch2.1.running_mean', 'features.5.banch2.1.running_var',
                        'features.5.banch2.1.num_batches_tracked', 'features.5.banch2.3.weight', 'features.5.banch2.4.weight', 'features.5.banch2.4.bias', 'features.5.banch2.4.running_mean', 'features.5.banch2.4.running_var',
                        'features.5.banch2.4.num_batches_tracked', 'features.5.banch2.5.weight', 'features.5.banch2.6.weight', 'features.5.banch2.6.bias', 'features.5.banch2.6.running_mean', 'features.5.banch2.6.running_var',
                        'features.5.banch2.6.num_batches_tracked', 'features.6.banch2.0.weight', 'features.6.banch2.1.weight', 'features.6.banch2.1.bias', 'features.6.banch2.1.running_mean', 'features.6.banch2.1.running_var',
                        'features.6.banch2.1.num_batches_tracked', 'features.6.banch2.3.weight', 'features.6.banch2.4.weight', 'features.6.banch2.4.bias', 'features.6.banch2.4.running_mean', 'features.6.banch2.4.running_var',
                        'features.6.banch2.4.num_batches_tracked', 'features.6.banch2.5.weight', 'features.6.banch2.6.weight', 'features.6.banch2.6.bias', 'features.6.banch2.6.running_mean', 'features.6.banch2.6.running_var',
                        'features.6.banch2.6.num_batches_tracked', 'features.7.banch2.0.weight', 'features.7.banch2.1.weight', 'features.7.banch2.1.bias', 'features.7.banch2.1.running_mean', 'features.7.banch2.1.running_var',
                        'features.7.banch2.1.num_batches_tracked', 'features.7.banch2.3.weight', 'features.7.banch2.4.weight', 'features.7.banch2.4.bias', 'features.7.banch2.4.running_mean', 'features.7.banch2.4.running_var',
                        'features.7.banch2.4.num_batches_tracked', 'features.7.banch2.5.weight', 'features.7.banch2.6.weight', 'features.7.banch2.6.bias', 'features.7.banch2.6.running_mean', 'features.7.banch2.6.running_var',
                        'features.7.banch2.6.num_batches_tracked', 'features.8.banch2.0.weight', 'features.8.banch2.1.weight', 'features.8.banch2.1.bias', 'features.8.banch2.1.running_mean', 'features.8.banch2.1.running_var',
                        'features.8.banch2.1.num_batches_tracked', 'features.8.banch2.3.weight', 'features.8.banch2.4.weight', 'features.8.banch2.4.bias', 'features.8.banch2.4.running_mean', 'features.8.banch2.4.running_var',
                        'features.8.banch2.4.num_batches_tracked', 'features.8.banch2.5.weight', 'features.8.banch2.6.weight', 'features.8.banch2.6.bias', 'features.8.banch2.6.running_mean', 'features.8.banch2.6.running_var',
                        'features.8.banch2.6.num_batches_tracked', 'features.9.banch2.0.weight', 'features.9.banch2.1.weight', 'features.9.banch2.1.bias', 'features.9.banch2.1.running_mean', 'features.9.banch2.1.running_var',
                        'features.9.banch2.1.num_batches_tracked', 'features.9.banch2.3.weight', 'features.9.banch2.4.weight', 'features.9.banch2.4.bias', 'features.9.banch2.4.running_mean', 'features.9.banch2.4.running_var',
                        'features.9.banch2.4.num_batches_tracked', 'features.9.banch2.5.weight', 'features.9.banch2.6.weight', 'features.9.banch2.6.bias', 'features.9.banch2.6.running_mean', 'features.9.banch2.6.running_var',
                        'features.9.banch2.6.num_batches_tracked', 'features.10.banch2.0.weight', 'features.10.banch2.1.weight', 'features.10.banch2.1.bias', 'features.10.banch2.1.running_mean', 'features.10.banch2.1.running_var',
                        'features.10.banch2.1.num_batches_tracked', 'features.10.banch2.3.weight', 'features.10.banch2.4.weight', 'features.10.banch2.4.bias', 'features.10.banch2.4.running_mean', 'features.10.banch2.4.running_var',
                        'features.10.banch2.4.num_batches_tracked', 'features.10.banch2.5.weight', 'features.10.banch2.6.weight', 'features.10.banch2.6.bias', 'features.10.banch2.6.running_mean', 'features.10.banch2.6.running_var',
                        'features.10.banch2.6.num_batches_tracked', 'features.11.banch2.0.weight', 'features.11.banch2.1.weight', 'features.11.banch2.1.bias', 'features.11.banch2.1.running_mean', 'features.11.banch2.1.running_var',
                        'features.11.banch2.1.num_batches_tracked', 'features.11.banch2.3.weight', 'features.11.banch2.4.weight', 'features.11.banch2.4.bias', 'features.11.banch2.4.running_mean', 'features.11.banch2.4.running_var',
                        'features.11.banch2.4.num_batches_tracked', 'features.11.banch2.5.weight', 'features.11.banch2.6.weight', 'features.11.banch2.6.bias', 'features.11.banch2.6.running_mean', 'features.11.banch2.6.running_var',
                        'features.11.banch2.6.num_batches_tracked', 'features.12.banch1.0.weight', 'features.12.banch1.1.weight', 'features.12.banch1.1.bias', 'features.12.banch1.1.running_mean', 'features.12.banch1.1.running_var',
                        'features.12.banch1.1.num_batches_tracked', 'features.12.banch1.2.weight', 'features.12.banch1.3.weight', 'features.12.banch1.3.bias', 'features.12.banch1.3.running_mean', 'features.12.banch1.3.running_var',
                        'features.12.banch1.3.num_batches_tracked', 'features.12.banch2.0.weight', 'features.12.banch2.1.weight', 'features.12.banch2.1.bias', 'features.12.banch2.1.running_mean', 'features.12.banch2.1.running_var',
                        'features.12.banch2.1.num_batches_tracked', 'features.12.banch2.3.weight', 'features.12.banch2.4.weight', 'features.12.banch2.4.bias', 'features.12.banch2.4.running_mean', 'features.12.banch2.4.running_var',
                        'features.12.banch2.4.num_batches_tracked', 'features.12.banch2.5.weight', 'features.12.banch2.6.weight', 'features.12.banch2.6.bias', 'features.12.banch2.6.running_mean', 'features.12.banch2.6.running_var',
                        'features.12.banch2.6.num_batches_tracked', 'features.13.banch2.0.weight', 'features.13.banch2.1.weight', 'features.13.banch2.1.bias', 'features.13.banch2.1.running_mean', 'features.13.banch2.1.running_var',
                        'features.13.banch2.1.num_batches_tracked', 'features.13.banch2.3.weight', 'features.13.banch2.4.weight', 'features.13.banch2.4.bias', 'features.13.banch2.4.running_mean', 'features.13.banch2.4.running_var',
                        'features.13.banch2.4.num_batches_tracked', 'features.13.banch2.5.weight', 'features.13.banch2.6.weight', 'features.13.banch2.6.bias', 'features.13.banch2.6.running_mean', 'features.13.banch2.6.running_var',
                        'features.13.banch2.6.num_batches_tracked', 'features.14.banch2.0.weight', 'features.14.banch2.1.weight', 'features.14.banch2.1.bias', 'features.14.banch2.1.running_mean', 'features.14.banch2.1.running_var',
                        'features.14.banch2.1.num_batches_tracked', 'features.14.banch2.3.weight', 'features.14.banch2.4.weight', 'features.14.banch2.4.bias', 'features.14.banch2.4.running_mean', 'features.14.banch2.4.running_var',
                        'features.14.banch2.4.num_batches_tracked', 'features.14.banch2.5.weight', 'features.14.banch2.6.weight', 'features.14.banch2.6.bias', 'features.14.banch2.6.running_mean', 'features.14.banch2.6.running_var',
                        'features.14.banch2.6.num_batches_tracked', 'features.15.banch2.0.weight', 'features.15.banch2.1.weight', 'features.15.banch2.1.bias', 'features.15.banch2.1.running_mean', 'features.15.banch2.1.running_var',
                        'features.15.banch2.1.num_batches_tracked', 'features.15.banch2.3.weight', 'features.15.banch2.4.weight', 'features.15.banch2.4.bias', 'features.15.banch2.4.running_mean', 'features.15.banch2.4.running_var',
                        'features.15.banch2.4.num_batches_tracked', 'features.15.banch2.5.weight', 'features.15.banch2.6.weight', 'features.15.banch2.6.bias', 'features.15.banch2.6.running_mean', 'features.15.banch2.6.running_var',
                        'features.15.banch2.6.num_batches_tracked', 'conv_last.0.weight', 'conv_last.1.weight', 'conv_last.1.bias', 'conv_last.1.running_mean', 'conv_last.1.running_var', 'conv_last.1.num_batches_tracked']
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

    def __load_pretrained_weights_shufflenet3d_v2_15(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'features.0.banch1.0.weight', 'features.0.banch1.1.weight', 'features.0.banch1.1.bias',
                        'features.0.banch1.1.running_mean', 'features.0.banch1.1.running_var', 'features.0.banch1.1.num_batches_tracked', 'features.0.banch1.2.weight', 'features.0.banch1.3.weight', 'features.0.banch1.3.bias', 'features.0.banch1.3.running_mean',
                        'features.0.banch1.3.running_var', 'features.0.banch1.3.num_batches_tracked', 'features.0.banch2.0.weight', 'features.0.banch2.1.weight', 'features.0.banch2.1.bias', 'features.0.banch2.1.running_mean', 'features.0.banch2.1.running_var',
                        'features.0.banch2.1.num_batches_tracked', 'features.0.banch2.3.weight', 'features.0.banch2.4.weight', 'features.0.banch2.4.bias', 'features.0.banch2.4.running_mean', 'features.0.banch2.4.running_var',
                        'features.0.banch2.4.num_batches_tracked', 'features.0.banch2.5.weight', 'features.0.banch2.6.weight', 'features.0.banch2.6.bias', 'features.0.banch2.6.running_mean', 'features.0.banch2.6.running_var',
                        'features.0.banch2.6.num_batches_tracked', 'features.1.banch2.0.weight', 'features.1.banch2.1.weight', 'features.1.banch2.1.bias', 'features.1.banch2.1.running_mean', 'features.1.banch2.1.running_var',
                        'features.1.banch2.1.num_batches_tracked', 'features.1.banch2.3.weight', 'features.1.banch2.4.weight', 'features.1.banch2.4.bias', 'features.1.banch2.4.running_mean', 'features.1.banch2.4.running_var',
                        'features.1.banch2.4.num_batches_tracked', 'features.1.banch2.5.weight', 'features.1.banch2.6.weight', 'features.1.banch2.6.bias', 'features.1.banch2.6.running_mean', 'features.1.banch2.6.running_var',
                        'features.1.banch2.6.num_batches_tracked', 'features.2.banch2.0.weight', 'features.2.banch2.1.weight', 'features.2.banch2.1.bias', 'features.2.banch2.1.running_mean', 'features.2.banch2.1.running_var',
                        'features.2.banch2.1.num_batches_tracked', 'features.2.banch2.3.weight', 'features.2.banch2.4.weight', 'features.2.banch2.4.bias', 'features.2.banch2.4.running_mean', 'features.2.banch2.4.running_var',
                        'features.2.banch2.4.num_batches_tracked', 'features.2.banch2.5.weight', 'features.2.banch2.6.weight', 'features.2.banch2.6.bias', 'features.2.banch2.6.running_mean', 'features.2.banch2.6.running_var',
                        'features.2.banch2.6.num_batches_tracked', 'features.3.banch2.0.weight', 'features.3.banch2.1.weight', 'features.3.banch2.1.bias', 'features.3.banch2.1.running_mean', 'features.3.banch2.1.running_var',
                        'features.3.banch2.1.num_batches_tracked', 'features.3.banch2.3.weight', 'features.3.banch2.4.weight', 'features.3.banch2.4.bias', 'features.3.banch2.4.running_mean', 'features.3.banch2.4.running_var',
                        'features.3.banch2.4.num_batches_tracked', 'features.3.banch2.5.weight', 'features.3.banch2.6.weight', 'features.3.banch2.6.bias', 'features.3.banch2.6.running_mean', 'features.3.banch2.6.running_var',
                        'features.3.banch2.6.num_batches_tracked', 'features.4.banch1.0.weight', 'features.4.banch1.1.weight', 'features.4.banch1.1.bias', 'features.4.banch1.1.running_mean', 'features.4.banch1.1.running_var',
                        'features.4.banch1.1.num_batches_tracked', 'features.4.banch1.2.weight', 'features.4.banch1.3.weight', 'features.4.banch1.3.bias', 'features.4.banch1.3.running_mean', 'features.4.banch1.3.running_var',
                        'features.4.banch1.3.num_batches_tracked', 'features.4.banch2.0.weight', 'features.4.banch2.1.weight', 'features.4.banch2.1.bias', 'features.4.banch2.1.running_mean', 'features.4.banch2.1.running_var',
                        'features.4.banch2.1.num_batches_tracked', 'features.4.banch2.3.weight', 'features.4.banch2.4.weight', 'features.4.banch2.4.bias', 'features.4.banch2.4.running_mean', 'features.4.banch2.4.running_var',
                        'features.4.banch2.4.num_batches_tracked', 'features.4.banch2.5.weight', 'features.4.banch2.6.weight', 'features.4.banch2.6.bias', 'features.4.banch2.6.running_mean', 'features.4.banch2.6.running_var',
                        'features.4.banch2.6.num_batches_tracked', 'features.5.banch2.0.weight', 'features.5.banch2.1.weight', 'features.5.banch2.1.bias', 'features.5.banch2.1.running_mean', 'features.5.banch2.1.running_var',
                        'features.5.banch2.1.num_batches_tracked', 'features.5.banch2.3.weight', 'features.5.banch2.4.weight', 'features.5.banch2.4.bias', 'features.5.banch2.4.running_mean', 'features.5.banch2.4.running_var',
                        'features.5.banch2.4.num_batches_tracked', 'features.5.banch2.5.weight', 'features.5.banch2.6.weight', 'features.5.banch2.6.bias', 'features.5.banch2.6.running_mean', 'features.5.banch2.6.running_var',
                        'features.5.banch2.6.num_batches_tracked', 'features.6.banch2.0.weight', 'features.6.banch2.1.weight', 'features.6.banch2.1.bias', 'features.6.banch2.1.running_mean', 'features.6.banch2.1.running_var',
                        'features.6.banch2.1.num_batches_tracked', 'features.6.banch2.3.weight', 'features.6.banch2.4.weight', 'features.6.banch2.4.bias', 'features.6.banch2.4.running_mean', 'features.6.banch2.4.running_var',
                        'features.6.banch2.4.num_batches_tracked', 'features.6.banch2.5.weight', 'features.6.banch2.6.weight', 'features.6.banch2.6.bias', 'features.6.banch2.6.running_mean', 'features.6.banch2.6.running_var',
                        'features.6.banch2.6.num_batches_tracked', 'features.7.banch2.0.weight', 'features.7.banch2.1.weight', 'features.7.banch2.1.bias', 'features.7.banch2.1.running_mean', 'features.7.banch2.1.running_var',
                        'features.7.banch2.1.num_batches_tracked', 'features.7.banch2.3.weight', 'features.7.banch2.4.weight', 'features.7.banch2.4.bias', 'features.7.banch2.4.running_mean', 'features.7.banch2.4.running_var',
                        'features.7.banch2.4.num_batches_tracked', 'features.7.banch2.5.weight', 'features.7.banch2.6.weight', 'features.7.banch2.6.bias', 'features.7.banch2.6.running_mean', 'features.7.banch2.6.running_var',
                        'features.7.banch2.6.num_batches_tracked', 'features.8.banch2.0.weight', 'features.8.banch2.1.weight', 'features.8.banch2.1.bias', 'features.8.banch2.1.running_mean', 'features.8.banch2.1.running_var',
                        'features.8.banch2.1.num_batches_tracked', 'features.8.banch2.3.weight', 'features.8.banch2.4.weight', 'features.8.banch2.4.bias', 'features.8.banch2.4.running_mean', 'features.8.banch2.4.running_var',
                        'features.8.banch2.4.num_batches_tracked', 'features.8.banch2.5.weight', 'features.8.banch2.6.weight', 'features.8.banch2.6.bias', 'features.8.banch2.6.running_mean', 'features.8.banch2.6.running_var',
                        'features.8.banch2.6.num_batches_tracked', 'features.9.banch2.0.weight', 'features.9.banch2.1.weight', 'features.9.banch2.1.bias', 'features.9.banch2.1.running_mean', 'features.9.banch2.1.running_var',
                        'features.9.banch2.1.num_batches_tracked', 'features.9.banch2.3.weight', 'features.9.banch2.4.weight', 'features.9.banch2.4.bias', 'features.9.banch2.4.running_mean', 'features.9.banch2.4.running_var',
                        'features.9.banch2.4.num_batches_tracked', 'features.9.banch2.5.weight', 'features.9.banch2.6.weight', 'features.9.banch2.6.bias', 'features.9.banch2.6.running_mean', 'features.9.banch2.6.running_var',
                        'features.9.banch2.6.num_batches_tracked', 'features.10.banch2.0.weight', 'features.10.banch2.1.weight', 'features.10.banch2.1.bias', 'features.10.banch2.1.running_mean', 'features.10.banch2.1.running_var',
                        'features.10.banch2.1.num_batches_tracked', 'features.10.banch2.3.weight', 'features.10.banch2.4.weight', 'features.10.banch2.4.bias', 'features.10.banch2.4.running_mean', 'features.10.banch2.4.running_var',
                        'features.10.banch2.4.num_batches_tracked', 'features.10.banch2.5.weight', 'features.10.banch2.6.weight', 'features.10.banch2.6.bias', 'features.10.banch2.6.running_mean', 'features.10.banch2.6.running_var',
                        'features.10.banch2.6.num_batches_tracked', 'features.11.banch2.0.weight', 'features.11.banch2.1.weight', 'features.11.banch2.1.bias', 'features.11.banch2.1.running_mean', 'features.11.banch2.1.running_var',
                        'features.11.banch2.1.num_batches_tracked', 'features.11.banch2.3.weight', 'features.11.banch2.4.weight', 'features.11.banch2.4.bias', 'features.11.banch2.4.running_mean', 'features.11.banch2.4.running_var',
                        'features.11.banch2.4.num_batches_tracked', 'features.11.banch2.5.weight', 'features.11.banch2.6.weight', 'features.11.banch2.6.bias', 'features.11.banch2.6.running_mean', 'features.11.banch2.6.running_var',
                        'features.11.banch2.6.num_batches_tracked', 'features.12.banch1.0.weight', 'features.12.banch1.1.weight', 'features.12.banch1.1.bias', 'features.12.banch1.1.running_mean', 'features.12.banch1.1.running_var',
                        'features.12.banch1.1.num_batches_tracked', 'features.12.banch1.2.weight', 'features.12.banch1.3.weight', 'features.12.banch1.3.bias', 'features.12.banch1.3.running_mean', 'features.12.banch1.3.running_var',
                        'features.12.banch1.3.num_batches_tracked', 'features.12.banch2.0.weight', 'features.12.banch2.1.weight', 'features.12.banch2.1.bias', 'features.12.banch2.1.running_mean', 'features.12.banch2.1.running_var',
                        'features.12.banch2.1.num_batches_tracked', 'features.12.banch2.3.weight', 'features.12.banch2.4.weight', 'features.12.banch2.4.bias', 'features.12.banch2.4.running_mean', 'features.12.banch2.4.running_var',
                        'features.12.banch2.4.num_batches_tracked', 'features.12.banch2.5.weight', 'features.12.banch2.6.weight', 'features.12.banch2.6.bias', 'features.12.banch2.6.running_mean', 'features.12.banch2.6.running_var',
                        'features.12.banch2.6.num_batches_tracked', 'features.13.banch2.0.weight', 'features.13.banch2.1.weight', 'features.13.banch2.1.bias', 'features.13.banch2.1.running_mean', 'features.13.banch2.1.running_var',
                        'features.13.banch2.1.num_batches_tracked', 'features.13.banch2.3.weight', 'features.13.banch2.4.weight', 'features.13.banch2.4.bias', 'features.13.banch2.4.running_mean', 'features.13.banch2.4.running_var',
                        'features.13.banch2.4.num_batches_tracked', 'features.13.banch2.5.weight', 'features.13.banch2.6.weight', 'features.13.banch2.6.bias', 'features.13.banch2.6.running_mean', 'features.13.banch2.6.running_var',
                        'features.13.banch2.6.num_batches_tracked', 'features.14.banch2.0.weight', 'features.14.banch2.1.weight', 'features.14.banch2.1.bias', 'features.14.banch2.1.running_mean', 'features.14.banch2.1.running_var',
                        'features.14.banch2.1.num_batches_tracked', 'features.14.banch2.3.weight', 'features.14.banch2.4.weight', 'features.14.banch2.4.bias', 'features.14.banch2.4.running_mean', 'features.14.banch2.4.running_var',
                        'features.14.banch2.4.num_batches_tracked', 'features.14.banch2.5.weight', 'features.14.banch2.6.weight', 'features.14.banch2.6.bias', 'features.14.banch2.6.running_mean', 'features.14.banch2.6.running_var',
                        'features.14.banch2.6.num_batches_tracked', 'features.15.banch2.0.weight', 'features.15.banch2.1.weight', 'features.15.banch2.1.bias', 'features.15.banch2.1.running_mean', 'features.15.banch2.1.running_var',
                        'features.15.banch2.1.num_batches_tracked', 'features.15.banch2.3.weight', 'features.15.banch2.4.weight', 'features.15.banch2.4.bias', 'features.15.banch2.4.running_mean', 'features.15.banch2.4.running_var',
                        'features.15.banch2.4.num_batches_tracked', 'features.15.banch2.5.weight', 'features.15.banch2.6.weight', 'features.15.banch2.6.bias', 'features.15.banch2.6.running_mean', 'features.15.banch2.6.running_var',
                        'features.15.banch2.6.num_batches_tracked', 'conv_last.0.weight', 'conv_last.1.weight', 'conv_last.1.bias', 'conv_last.1.running_mean', 'conv_last.1.running_var', 'conv_last.1.num_batches_tracked']
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

    def __load_pretrained_weights_shufflenet3d_v2_20(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'features.0.banch1.0.weight', 'features.0.banch1.1.weight', 'features.0.banch1.1.bias',
                        'features.0.banch1.1.running_mean', 'features.0.banch1.1.running_var', 'features.0.banch1.1.num_batches_tracked', 'features.0.banch1.2.weight', 'features.0.banch1.3.weight', 'features.0.banch1.3.bias', 'features.0.banch1.3.running_mean',
                        'features.0.banch1.3.running_var', 'features.0.banch1.3.num_batches_tracked', 'features.0.banch2.0.weight', 'features.0.banch2.1.weight', 'features.0.banch2.1.bias', 'features.0.banch2.1.running_mean', 'features.0.banch2.1.running_var',
                        'features.0.banch2.1.num_batches_tracked', 'features.0.banch2.3.weight', 'features.0.banch2.4.weight', 'features.0.banch2.4.bias', 'features.0.banch2.4.running_mean', 'features.0.banch2.4.running_var',
                        'features.0.banch2.4.num_batches_tracked', 'features.0.banch2.5.weight', 'features.0.banch2.6.weight', 'features.0.banch2.6.bias', 'features.0.banch2.6.running_mean', 'features.0.banch2.6.running_var',
                        'features.0.banch2.6.num_batches_tracked', 'features.1.banch2.0.weight', 'features.1.banch2.1.weight', 'features.1.banch2.1.bias', 'features.1.banch2.1.running_mean', 'features.1.banch2.1.running_var',
                        'features.1.banch2.1.num_batches_tracked', 'features.1.banch2.3.weight', 'features.1.banch2.4.weight', 'features.1.banch2.4.bias', 'features.1.banch2.4.running_mean', 'features.1.banch2.4.running_var',
                        'features.1.banch2.4.num_batches_tracked', 'features.1.banch2.5.weight', 'features.1.banch2.6.weight', 'features.1.banch2.6.bias', 'features.1.banch2.6.running_mean', 'features.1.banch2.6.running_var',
                        'features.1.banch2.6.num_batches_tracked', 'features.2.banch2.0.weight', 'features.2.banch2.1.weight', 'features.2.banch2.1.bias', 'features.2.banch2.1.running_mean', 'features.2.banch2.1.running_var',
                        'features.2.banch2.1.num_batches_tracked', 'features.2.banch2.3.weight', 'features.2.banch2.4.weight', 'features.2.banch2.4.bias', 'features.2.banch2.4.running_mean', 'features.2.banch2.4.running_var',
                        'features.2.banch2.4.num_batches_tracked', 'features.2.banch2.5.weight', 'features.2.banch2.6.weight', 'features.2.banch2.6.bias', 'features.2.banch2.6.running_mean', 'features.2.banch2.6.running_var',
                        'features.2.banch2.6.num_batches_tracked', 'features.3.banch2.0.weight', 'features.3.banch2.1.weight', 'features.3.banch2.1.bias', 'features.3.banch2.1.running_mean', 'features.3.banch2.1.running_var',
                        'features.3.banch2.1.num_batches_tracked', 'features.3.banch2.3.weight', 'features.3.banch2.4.weight', 'features.3.banch2.4.bias', 'features.3.banch2.4.running_mean', 'features.3.banch2.4.running_var',
                        'features.3.banch2.4.num_batches_tracked', 'features.3.banch2.5.weight', 'features.3.banch2.6.weight', 'features.3.banch2.6.bias', 'features.3.banch2.6.running_mean', 'features.3.banch2.6.running_var',
                        'features.3.banch2.6.num_batches_tracked', 'features.4.banch1.0.weight', 'features.4.banch1.1.weight', 'features.4.banch1.1.bias', 'features.4.banch1.1.running_mean', 'features.4.banch1.1.running_var',
                        'features.4.banch1.1.num_batches_tracked', 'features.4.banch1.2.weight', 'features.4.banch1.3.weight', 'features.4.banch1.3.bias', 'features.4.banch1.3.running_mean', 'features.4.banch1.3.running_var',
                        'features.4.banch1.3.num_batches_tracked', 'features.4.banch2.0.weight', 'features.4.banch2.1.weight', 'features.4.banch2.1.bias', 'features.4.banch2.1.running_mean', 'features.4.banch2.1.running_var',
                        'features.4.banch2.1.num_batches_tracked', 'features.4.banch2.3.weight', 'features.4.banch2.4.weight', 'features.4.banch2.4.bias', 'features.4.banch2.4.running_mean', 'features.4.banch2.4.running_var',
                        'features.4.banch2.4.num_batches_tracked', 'features.4.banch2.5.weight', 'features.4.banch2.6.weight', 'features.4.banch2.6.bias', 'features.4.banch2.6.running_mean', 'features.4.banch2.6.running_var',
                        'features.4.banch2.6.num_batches_tracked', 'features.5.banch2.0.weight', 'features.5.banch2.1.weight', 'features.5.banch2.1.bias', 'features.5.banch2.1.running_mean', 'features.5.banch2.1.running_var',
                        'features.5.banch2.1.num_batches_tracked', 'features.5.banch2.3.weight', 'features.5.banch2.4.weight', 'features.5.banch2.4.bias', 'features.5.banch2.4.running_mean', 'features.5.banch2.4.running_var',
                        'features.5.banch2.4.num_batches_tracked', 'features.5.banch2.5.weight', 'features.5.banch2.6.weight', 'features.5.banch2.6.bias', 'features.5.banch2.6.running_mean', 'features.5.banch2.6.running_var',
                        'features.5.banch2.6.num_batches_tracked', 'features.6.banch2.0.weight', 'features.6.banch2.1.weight', 'features.6.banch2.1.bias', 'features.6.banch2.1.running_mean', 'features.6.banch2.1.running_var',
                        'features.6.banch2.1.num_batches_tracked', 'features.6.banch2.3.weight', 'features.6.banch2.4.weight', 'features.6.banch2.4.bias', 'features.6.banch2.4.running_mean', 'features.6.banch2.4.running_var',
                        'features.6.banch2.4.num_batches_tracked', 'features.6.banch2.5.weight', 'features.6.banch2.6.weight', 'features.6.banch2.6.bias', 'features.6.banch2.6.running_mean', 'features.6.banch2.6.running_var',
                        'features.6.banch2.6.num_batches_tracked', 'features.7.banch2.0.weight', 'features.7.banch2.1.weight', 'features.7.banch2.1.bias', 'features.7.banch2.1.running_mean', 'features.7.banch2.1.running_var',
                        'features.7.banch2.1.num_batches_tracked', 'features.7.banch2.3.weight', 'features.7.banch2.4.weight', 'features.7.banch2.4.bias', 'features.7.banch2.4.running_mean', 'features.7.banch2.4.running_var',
                        'features.7.banch2.4.num_batches_tracked', 'features.7.banch2.5.weight', 'features.7.banch2.6.weight', 'features.7.banch2.6.bias', 'features.7.banch2.6.running_mean', 'features.7.banch2.6.running_var',
                        'features.7.banch2.6.num_batches_tracked', 'features.8.banch2.0.weight', 'features.8.banch2.1.weight', 'features.8.banch2.1.bias', 'features.8.banch2.1.running_mean', 'features.8.banch2.1.running_var',
                        'features.8.banch2.1.num_batches_tracked', 'features.8.banch2.3.weight', 'features.8.banch2.4.weight', 'features.8.banch2.4.bias', 'features.8.banch2.4.running_mean', 'features.8.banch2.4.running_var',
                        'features.8.banch2.4.num_batches_tracked', 'features.8.banch2.5.weight', 'features.8.banch2.6.weight', 'features.8.banch2.6.bias', 'features.8.banch2.6.running_mean', 'features.8.banch2.6.running_var',
                        'features.8.banch2.6.num_batches_tracked', 'features.9.banch2.0.weight', 'features.9.banch2.1.weight', 'features.9.banch2.1.bias', 'features.9.banch2.1.running_mean', 'features.9.banch2.1.running_var',
                        'features.9.banch2.1.num_batches_tracked', 'features.9.banch2.3.weight', 'features.9.banch2.4.weight', 'features.9.banch2.4.bias', 'features.9.banch2.4.running_mean', 'features.9.banch2.4.running_var',
                        'features.9.banch2.4.num_batches_tracked', 'features.9.banch2.5.weight', 'features.9.banch2.6.weight', 'features.9.banch2.6.bias', 'features.9.banch2.6.running_mean', 'features.9.banch2.6.running_var',
                        'features.9.banch2.6.num_batches_tracked', 'features.10.banch2.0.weight', 'features.10.banch2.1.weight', 'features.10.banch2.1.bias', 'features.10.banch2.1.running_mean', 'features.10.banch2.1.running_var',
                        'features.10.banch2.1.num_batches_tracked', 'features.10.banch2.3.weight', 'features.10.banch2.4.weight', 'features.10.banch2.4.bias', 'features.10.banch2.4.running_mean', 'features.10.banch2.4.running_var',
                        'features.10.banch2.4.num_batches_tracked', 'features.10.banch2.5.weight', 'features.10.banch2.6.weight', 'features.10.banch2.6.bias', 'features.10.banch2.6.running_mean', 'features.10.banch2.6.running_var',
                        'features.10.banch2.6.num_batches_tracked', 'features.11.banch2.0.weight', 'features.11.banch2.1.weight', 'features.11.banch2.1.bias', 'features.11.banch2.1.running_mean', 'features.11.banch2.1.running_var',
                        'features.11.banch2.1.num_batches_tracked', 'features.11.banch2.3.weight', 'features.11.banch2.4.weight', 'features.11.banch2.4.bias', 'features.11.banch2.4.running_mean', 'features.11.banch2.4.running_var',
                        'features.11.banch2.4.num_batches_tracked', 'features.11.banch2.5.weight', 'features.11.banch2.6.weight', 'features.11.banch2.6.bias', 'features.11.banch2.6.running_mean', 'features.11.banch2.6.running_var',
                        'features.11.banch2.6.num_batches_tracked', 'features.12.banch1.0.weight', 'features.12.banch1.1.weight', 'features.12.banch1.1.bias', 'features.12.banch1.1.running_mean', 'features.12.banch1.1.running_var',
                        'features.12.banch1.1.num_batches_tracked', 'features.12.banch1.2.weight', 'features.12.banch1.3.weight', 'features.12.banch1.3.bias', 'features.12.banch1.3.running_mean', 'features.12.banch1.3.running_var',
                        'features.12.banch1.3.num_batches_tracked', 'features.12.banch2.0.weight', 'features.12.banch2.1.weight', 'features.12.banch2.1.bias', 'features.12.banch2.1.running_mean', 'features.12.banch2.1.running_var',
                        'features.12.banch2.1.num_batches_tracked', 'features.12.banch2.3.weight', 'features.12.banch2.4.weight', 'features.12.banch2.4.bias', 'features.12.banch2.4.running_mean', 'features.12.banch2.4.running_var',
                        'features.12.banch2.4.num_batches_tracked', 'features.12.banch2.5.weight', 'features.12.banch2.6.weight', 'features.12.banch2.6.bias', 'features.12.banch2.6.running_mean', 'features.12.banch2.6.running_var',
                        'features.12.banch2.6.num_batches_tracked', 'features.13.banch2.0.weight', 'features.13.banch2.1.weight', 'features.13.banch2.1.bias', 'features.13.banch2.1.running_mean', 'features.13.banch2.1.running_var',
                        'features.13.banch2.1.num_batches_tracked', 'features.13.banch2.3.weight', 'features.13.banch2.4.weight', 'features.13.banch2.4.bias', 'features.13.banch2.4.running_mean', 'features.13.banch2.4.running_var',
                        'features.13.banch2.4.num_batches_tracked', 'features.13.banch2.5.weight', 'features.13.banch2.6.weight', 'features.13.banch2.6.bias', 'features.13.banch2.6.running_mean', 'features.13.banch2.6.running_var',
                        'features.13.banch2.6.num_batches_tracked', 'features.14.banch2.0.weight', 'features.14.banch2.1.weight', 'features.14.banch2.1.bias', 'features.14.banch2.1.running_mean', 'features.14.banch2.1.running_var',
                        'features.14.banch2.1.num_batches_tracked', 'features.14.banch2.3.weight', 'features.14.banch2.4.weight', 'features.14.banch2.4.bias', 'features.14.banch2.4.running_mean', 'features.14.banch2.4.running_var',
                        'features.14.banch2.4.num_batches_tracked', 'features.14.banch2.5.weight', 'features.14.banch2.6.weight', 'features.14.banch2.6.bias', 'features.14.banch2.6.running_mean', 'features.14.banch2.6.running_var',
                        'features.14.banch2.6.num_batches_tracked', 'features.15.banch2.0.weight', 'features.15.banch2.1.weight', 'features.15.banch2.1.bias', 'features.15.banch2.1.running_mean', 'features.15.banch2.1.running_var',
                        'features.15.banch2.1.num_batches_tracked', 'features.15.banch2.3.weight', 'features.15.banch2.4.weight', 'features.15.banch2.4.bias', 'features.15.banch2.4.running_mean', 'features.15.banch2.4.running_var',
                        'features.15.banch2.4.num_batches_tracked', 'features.15.banch2.5.weight', 'features.15.banch2.6.weight', 'features.15.banch2.6.bias', 'features.15.banch2.6.running_mean', 'features.15.banch2.6.running_var',
                        'features.15.banch2.6.num_batches_tracked', 'conv_last.0.weight', 'conv_last.1.weight', 'conv_last.1.bias', 'conv_last.1.running_mean', 'conv_last.1.running_var', 'conv_last.1.num_batches_tracked']
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
    model = ShuffleNet3D_v2(num_classes=101, width_mult=1.0, pretrained='pretrained/pretrained_shufflenet3D_v2/kinetics_shufflenetv2_1.0x_RGB_16_best.pth')
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)