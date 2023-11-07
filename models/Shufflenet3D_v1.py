# https://github.com/okankop/Efficient-3DCNNs/blob/master/models/shufflenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False), nn.BatchNorm3d(oup), nn.ReLU(inplace=True))

def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.groups = groups
        mid_planes = out_planes//4
        if self.stride == 2:
            out_planes = out_planes - in_planes
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 2:
            self.shortcut = nn.AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.stride == 2:
            out = self.relu(torch.cat([out, self.shortcut(x)], 1))
        else:
            out = self.relu(out + x)
        return out

class ShuffleNet3D_v1(nn.Module):
    def __init__(self, groups, width_mult=1, num_classes=400, pretrained=None):
        super(ShuffleNet3D_v1, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.groups = groups
        num_blocks = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            out_planes = [24, 144, 288, 567]
        elif groups == 2:
            out_planes = [24, 200, 400, 800]
        elif groups == 3:
            out_planes = [24, 240, 480, 960]
        elif groups == 4:
            out_planes = [24, 272, 544, 1088]
        elif groups == 8:
            out_planes = [24, 384, 768, 1536]
        else:
            raise ValueError("""{} groups is not supported for 1x1 Grouped Convolutions""".format(groups))
        out_planes = [int(i * width_mult) for i in out_planes]
        self.in_planes = out_planes[0]
        self.conv1 = conv_bn(3, self.in_planes, stride=(1, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(out_planes[1], num_blocks[0], self.groups)
        self.layer2 = self._make_layer(out_planes[2], num_blocks[1], self.groups)
        self.layer3 = self._make_layer(out_planes[3], num_blocks[2], self.groups)
        # building classifier
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(out_planes[3], self.num_classes))
        if pretrained:
            if width_mult == 0.5:
                print('Loading pretrained weights for shufflenet3D_v1 with width = 0.5...')
                self.__load_pretrained_weights_shufflenet3d_v1_05()
            elif width_mult == 1.0:
                print('Loading pretrained weights for shufflenet3D_v1 with width = 1.0...')
                self.__load_pretrained_weights_shufflenet3d_v1_10()
            elif width_mult == 1.5:
                print('Loading pretrained weights for shufflenet3D_v1 with width = 1.5...')
                self.__load_pretrained_weights_shufflenet3d_v1_15()
            elif width_mult == 2.0:
                print('Loading pretrained weights for shufflenet3D_v1 with width = 2.0...')
                self.__load_pretrained_weights_shufflenet3d_v1_20()

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, out_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def __load_pretrained_weights_shufflenet3d_v1_05(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
                        'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
                        'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
                        'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
                        'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight',
                        'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean',
                        'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked',
                        'layer1.3.conv1.weight', 'layer1.3.bn1.weight', 'layer1.3.bn1.bias', 'layer1.3.bn1.running_mean', 'layer1.3.bn1.running_var', 'layer1.3.bn1.num_batches_tracked', 'layer1.3.conv2.weight', 'layer1.3.bn2.weight', 'layer1.3.bn2.bias',
                        'layer1.3.bn2.running_mean', 'layer1.3.bn2.running_var', 'layer1.3.bn2.num_batches_tracked', 'layer1.3.conv3.weight', 'layer1.3.bn3.weight', 'layer1.3.bn3.bias', 'layer1.3.bn3.running_mean', 'layer1.3.bn3.running_var',
                        'layer1.3.bn3.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight',
                        'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean',
                        'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked',
                        'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias',
                        'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var',
                        'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight',
                        'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean',
                        'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked',
                        'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked', 'layer2.4.conv1.weight', 'layer2.4.bn1.weight', 'layer2.4.bn1.bias',
                        'layer2.4.bn1.running_mean', 'layer2.4.bn1.running_var', 'layer2.4.bn1.num_batches_tracked', 'layer2.4.conv2.weight', 'layer2.4.bn2.weight', 'layer2.4.bn2.bias', 'layer2.4.bn2.running_mean', 'layer2.4.bn2.running_var',
                        'layer2.4.bn2.num_batches_tracked', 'layer2.4.conv3.weight', 'layer2.4.bn3.weight', 'layer2.4.bn3.bias', 'layer2.4.bn3.running_mean', 'layer2.4.bn3.running_var', 'layer2.4.bn3.num_batches_tracked', 'layer2.5.conv1.weight',
                        'layer2.5.bn1.weight', 'layer2.5.bn1.bias', 'layer2.5.bn1.running_mean', 'layer2.5.bn1.running_var', 'layer2.5.bn1.num_batches_tracked', 'layer2.5.conv2.weight', 'layer2.5.bn2.weight', 'layer2.5.bn2.bias', 'layer2.5.bn2.running_mean',
                        'layer2.5.bn2.running_var', 'layer2.5.bn2.num_batches_tracked', 'layer2.5.conv3.weight', 'layer2.5.bn3.weight', 'layer2.5.bn3.bias', 'layer2.5.bn3.running_mean', 'layer2.5.bn3.running_var', 'layer2.5.bn3.num_batches_tracked',
                        'layer2.6.conv1.weight', 'layer2.6.bn1.weight', 'layer2.6.bn1.bias', 'layer2.6.bn1.running_mean', 'layer2.6.bn1.running_var', 'layer2.6.bn1.num_batches_tracked', 'layer2.6.conv2.weight', 'layer2.6.bn2.weight', 'layer2.6.bn2.bias',
                        'layer2.6.bn2.running_mean', 'layer2.6.bn2.running_var', 'layer2.6.bn2.num_batches_tracked', 'layer2.6.conv3.weight', 'layer2.6.bn3.weight', 'layer2.6.bn3.bias', 'layer2.6.bn3.running_mean', 'layer2.6.bn3.running_var',
                        'layer2.6.bn3.num_batches_tracked', 'layer2.7.conv1.weight', 'layer2.7.bn1.weight', 'layer2.7.bn1.bias', 'layer2.7.bn1.running_mean', 'layer2.7.bn1.running_var', 'layer2.7.bn1.num_batches_tracked', 'layer2.7.conv2.weight',
                        'layer2.7.bn2.weight', 'layer2.7.bn2.bias', 'layer2.7.bn2.running_mean', 'layer2.7.bn2.running_var', 'layer2.7.bn2.num_batches_tracked', 'layer2.7.conv3.weight', 'layer2.7.bn3.weight', 'layer2.7.bn3.bias', 'layer2.7.bn3.running_mean',
                        'layer2.7.bn3.running_var', 'layer2.7.bn3.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked',
                        'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias',
                        'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var',
                        'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight',
                        'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean',
                        'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked',
                        'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
                        'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
                        'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked']
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

    def __load_pretrained_weights_shufflenet3d_v1_10(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
                        'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
                        'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
                        'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
                        'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight',
                        'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean',
                        'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked',
                        'layer1.3.conv1.weight', 'layer1.3.bn1.weight', 'layer1.3.bn1.bias', 'layer1.3.bn1.running_mean', 'layer1.3.bn1.running_var', 'layer1.3.bn1.num_batches_tracked', 'layer1.3.conv2.weight', 'layer1.3.bn2.weight', 'layer1.3.bn2.bias',
                        'layer1.3.bn2.running_mean', 'layer1.3.bn2.running_var', 'layer1.3.bn2.num_batches_tracked', 'layer1.3.conv3.weight', 'layer1.3.bn3.weight', 'layer1.3.bn3.bias', 'layer1.3.bn3.running_mean', 'layer1.3.bn3.running_var',
                        'layer1.3.bn3.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight',
                        'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean',
                        'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked',
                        'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias',
                        'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var',
                        'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight',
                        'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean',
                        'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked',
                        'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked', 'layer2.4.conv1.weight', 'layer2.4.bn1.weight', 'layer2.4.bn1.bias',
                        'layer2.4.bn1.running_mean', 'layer2.4.bn1.running_var', 'layer2.4.bn1.num_batches_tracked', 'layer2.4.conv2.weight', 'layer2.4.bn2.weight', 'layer2.4.bn2.bias', 'layer2.4.bn2.running_mean', 'layer2.4.bn2.running_var',
                        'layer2.4.bn2.num_batches_tracked', 'layer2.4.conv3.weight', 'layer2.4.bn3.weight', 'layer2.4.bn3.bias', 'layer2.4.bn3.running_mean', 'layer2.4.bn3.running_var', 'layer2.4.bn3.num_batches_tracked', 'layer2.5.conv1.weight',
                        'layer2.5.bn1.weight', 'layer2.5.bn1.bias', 'layer2.5.bn1.running_mean', 'layer2.5.bn1.running_var', 'layer2.5.bn1.num_batches_tracked', 'layer2.5.conv2.weight', 'layer2.5.bn2.weight', 'layer2.5.bn2.bias', 'layer2.5.bn2.running_mean',
                        'layer2.5.bn2.running_var', 'layer2.5.bn2.num_batches_tracked', 'layer2.5.conv3.weight', 'layer2.5.bn3.weight', 'layer2.5.bn3.bias', 'layer2.5.bn3.running_mean', 'layer2.5.bn3.running_var', 'layer2.5.bn3.num_batches_tracked',
                        'layer2.6.conv1.weight', 'layer2.6.bn1.weight', 'layer2.6.bn1.bias', 'layer2.6.bn1.running_mean', 'layer2.6.bn1.running_var', 'layer2.6.bn1.num_batches_tracked', 'layer2.6.conv2.weight', 'layer2.6.bn2.weight', 'layer2.6.bn2.bias',
                        'layer2.6.bn2.running_mean', 'layer2.6.bn2.running_var', 'layer2.6.bn2.num_batches_tracked', 'layer2.6.conv3.weight', 'layer2.6.bn3.weight', 'layer2.6.bn3.bias', 'layer2.6.bn3.running_mean', 'layer2.6.bn3.running_var',
                        'layer2.6.bn3.num_batches_tracked', 'layer2.7.conv1.weight', 'layer2.7.bn1.weight', 'layer2.7.bn1.bias', 'layer2.7.bn1.running_mean', 'layer2.7.bn1.running_var', 'layer2.7.bn1.num_batches_tracked', 'layer2.7.conv2.weight',
                        'layer2.7.bn2.weight', 'layer2.7.bn2.bias', 'layer2.7.bn2.running_mean', 'layer2.7.bn2.running_var', 'layer2.7.bn2.num_batches_tracked', 'layer2.7.conv3.weight', 'layer2.7.bn3.weight', 'layer2.7.bn3.bias', 'layer2.7.bn3.running_mean',
                        'layer2.7.bn3.running_var', 'layer2.7.bn3.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked',
                        'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias',
                        'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var',
                        'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight',
                        'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean',
                        'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked',
                        'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
                        'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
                        'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked']
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

    def __load_pretrained_weights_shufflenet3d_v1_15(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
                        'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
                        'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
                        'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
                        'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight',
                        'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean',
                        'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked',
                        'layer1.3.conv1.weight', 'layer1.3.bn1.weight', 'layer1.3.bn1.bias', 'layer1.3.bn1.running_mean', 'layer1.3.bn1.running_var', 'layer1.3.bn1.num_batches_tracked', 'layer1.3.conv2.weight', 'layer1.3.bn2.weight', 'layer1.3.bn2.bias',
                        'layer1.3.bn2.running_mean', 'layer1.3.bn2.running_var', 'layer1.3.bn2.num_batches_tracked', 'layer1.3.conv3.weight', 'layer1.3.bn3.weight', 'layer1.3.bn3.bias', 'layer1.3.bn3.running_mean', 'layer1.3.bn3.running_var',
                        'layer1.3.bn3.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight',
                        'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean',
                        'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked',
                        'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias',
                        'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var',
                        'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight',
                        'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean',
                        'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked',
                        'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked', 'layer2.4.conv1.weight', 'layer2.4.bn1.weight', 'layer2.4.bn1.bias',
                        'layer2.4.bn1.running_mean', 'layer2.4.bn1.running_var', 'layer2.4.bn1.num_batches_tracked', 'layer2.4.conv2.weight', 'layer2.4.bn2.weight', 'layer2.4.bn2.bias', 'layer2.4.bn2.running_mean', 'layer2.4.bn2.running_var',
                        'layer2.4.bn2.num_batches_tracked', 'layer2.4.conv3.weight', 'layer2.4.bn3.weight', 'layer2.4.bn3.bias', 'layer2.4.bn3.running_mean', 'layer2.4.bn3.running_var', 'layer2.4.bn3.num_batches_tracked', 'layer2.5.conv1.weight',
                        'layer2.5.bn1.weight', 'layer2.5.bn1.bias', 'layer2.5.bn1.running_mean', 'layer2.5.bn1.running_var', 'layer2.5.bn1.num_batches_tracked', 'layer2.5.conv2.weight', 'layer2.5.bn2.weight', 'layer2.5.bn2.bias', 'layer2.5.bn2.running_mean',
                        'layer2.5.bn2.running_var', 'layer2.5.bn2.num_batches_tracked', 'layer2.5.conv3.weight', 'layer2.5.bn3.weight', 'layer2.5.bn3.bias', 'layer2.5.bn3.running_mean', 'layer2.5.bn3.running_var', 'layer2.5.bn3.num_batches_tracked',
                        'layer2.6.conv1.weight', 'layer2.6.bn1.weight', 'layer2.6.bn1.bias', 'layer2.6.bn1.running_mean', 'layer2.6.bn1.running_var', 'layer2.6.bn1.num_batches_tracked', 'layer2.6.conv2.weight', 'layer2.6.bn2.weight', 'layer2.6.bn2.bias',
                        'layer2.6.bn2.running_mean', 'layer2.6.bn2.running_var', 'layer2.6.bn2.num_batches_tracked', 'layer2.6.conv3.weight', 'layer2.6.bn3.weight', 'layer2.6.bn3.bias', 'layer2.6.bn3.running_mean', 'layer2.6.bn3.running_var',
                        'layer2.6.bn3.num_batches_tracked', 'layer2.7.conv1.weight', 'layer2.7.bn1.weight', 'layer2.7.bn1.bias', 'layer2.7.bn1.running_mean', 'layer2.7.bn1.running_var', 'layer2.7.bn1.num_batches_tracked', 'layer2.7.conv2.weight',
                        'layer2.7.bn2.weight', 'layer2.7.bn2.bias', 'layer2.7.bn2.running_mean', 'layer2.7.bn2.running_var', 'layer2.7.bn2.num_batches_tracked', 'layer2.7.conv3.weight', 'layer2.7.bn3.weight', 'layer2.7.bn3.bias', 'layer2.7.bn3.running_mean',
                        'layer2.7.bn3.running_var', 'layer2.7.bn3.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked',
                        'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias',
                        'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var',
                        'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight',
                        'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean',
                        'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked',
                        'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
                        'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
                        'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked']
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

    def __load_pretrained_weights_shufflenet3d_v1_20(self):
        corresp_name = ['conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
                        'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
                        'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
                        'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
                        'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight',
                        'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean',
                        'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked',
                        'layer1.3.conv1.weight', 'layer1.3.bn1.weight', 'layer1.3.bn1.bias', 'layer1.3.bn1.running_mean', 'layer1.3.bn1.running_var', 'layer1.3.bn1.num_batches_tracked', 'layer1.3.conv2.weight', 'layer1.3.bn2.weight', 'layer1.3.bn2.bias',
                        'layer1.3.bn2.running_mean', 'layer1.3.bn2.running_var', 'layer1.3.bn2.num_batches_tracked', 'layer1.3.conv3.weight', 'layer1.3.bn3.weight', 'layer1.3.bn3.bias', 'layer1.3.bn3.running_mean', 'layer1.3.bn3.running_var',
                        'layer1.3.bn3.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight',
                        'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean',
                        'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked',
                        'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias',
                        'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var',
                        'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight',
                        'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean',
                        'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked',
                        'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked', 'layer2.4.conv1.weight', 'layer2.4.bn1.weight', 'layer2.4.bn1.bias',
                        'layer2.4.bn1.running_mean', 'layer2.4.bn1.running_var', 'layer2.4.bn1.num_batches_tracked', 'layer2.4.conv2.weight', 'layer2.4.bn2.weight', 'layer2.4.bn2.bias', 'layer2.4.bn2.running_mean', 'layer2.4.bn2.running_var',
                        'layer2.4.bn2.num_batches_tracked', 'layer2.4.conv3.weight', 'layer2.4.bn3.weight', 'layer2.4.bn3.bias', 'layer2.4.bn3.running_mean', 'layer2.4.bn3.running_var', 'layer2.4.bn3.num_batches_tracked', 'layer2.5.conv1.weight',
                        'layer2.5.bn1.weight', 'layer2.5.bn1.bias', 'layer2.5.bn1.running_mean', 'layer2.5.bn1.running_var', 'layer2.5.bn1.num_batches_tracked', 'layer2.5.conv2.weight', 'layer2.5.bn2.weight', 'layer2.5.bn2.bias', 'layer2.5.bn2.running_mean',
                        'layer2.5.bn2.running_var', 'layer2.5.bn2.num_batches_tracked', 'layer2.5.conv3.weight', 'layer2.5.bn3.weight', 'layer2.5.bn3.bias', 'layer2.5.bn3.running_mean', 'layer2.5.bn3.running_var', 'layer2.5.bn3.num_batches_tracked',
                        'layer2.6.conv1.weight', 'layer2.6.bn1.weight', 'layer2.6.bn1.bias', 'layer2.6.bn1.running_mean', 'layer2.6.bn1.running_var', 'layer2.6.bn1.num_batches_tracked', 'layer2.6.conv2.weight', 'layer2.6.bn2.weight', 'layer2.6.bn2.bias',
                        'layer2.6.bn2.running_mean', 'layer2.6.bn2.running_var', 'layer2.6.bn2.num_batches_tracked', 'layer2.6.conv3.weight', 'layer2.6.bn3.weight', 'layer2.6.bn3.bias', 'layer2.6.bn3.running_mean', 'layer2.6.bn3.running_var',
                        'layer2.6.bn3.num_batches_tracked', 'layer2.7.conv1.weight', 'layer2.7.bn1.weight', 'layer2.7.bn1.bias', 'layer2.7.bn1.running_mean', 'layer2.7.bn1.running_var', 'layer2.7.bn1.num_batches_tracked', 'layer2.7.conv2.weight',
                        'layer2.7.bn2.weight', 'layer2.7.bn2.bias', 'layer2.7.bn2.running_mean', 'layer2.7.bn2.running_var', 'layer2.7.bn2.num_batches_tracked', 'layer2.7.conv3.weight', 'layer2.7.bn3.weight', 'layer2.7.bn3.bias', 'layer2.7.bn3.running_mean',
                        'layer2.7.bn3.running_var', 'layer2.7.bn3.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked',
                        'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias',
                        'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var',
                        'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight',
                        'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean',
                        'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked',
                        'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
                        'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
                        'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked']
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
    model = ShuffleNet3D_v1(groups=3, num_classes=101, width_mult=2, pretrained='pretrained/pretrained_shufflenet3D_v1/kinetics_shufflenet_2.0x_G3_RGB_16_best.pth')
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)