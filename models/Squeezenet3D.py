# https://github.com/okankop/Efficient-3DCNNs/blob/master/models/squeezenet.py
import math
import torch
import torch.nn as nn
from collections import OrderedDict

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)
        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)
        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)
        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
            out += x
        out = self.relu(out)
        return out

class SqueezeNet3D(nn.Module):
    def __init__(self, sample_size, sample_duration, version=1.1, num_classes=600, pretrained=None):
        super(SqueezeNet3D, self).__init__()
        self.pretrained = pretrained
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:" "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv3d(3, 96, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3)),
                                          nn.BatchNorm3d(96),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(96, 16, 64, 64),
                                          Fire(128, 16, 64, 64, use_bypass=True),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(128, 32, 128, 128),
                                          Fire(256, 32, 128, 128, use_bypass=True),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(256, 48, 192, 192),
                                          Fire(384, 48, 192, 192, use_bypass=True),
                                          Fire(384, 64, 256, 256),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(512, 64, 256, 256, use_bypass=True))
        if version == 1.1:
            self.features = nn.Sequential(nn.Conv3d(3, 64, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)),
                                          nn.BatchNorm3d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(64, 16, 64, 64),
                                          Fire(128, 16, 64, 64, use_bypass=True),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(128, 32, 128, 128),
                                          Fire(256, 32, 128, 128, use_bypass=True),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(256, 48, 192, 192),
                                          Fire(384, 48, 192, 192, use_bypass=True),
                                          nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                                          Fire(384, 64, 256, 256),
                                          Fire(512, 64, 256, 256, use_bypass=True))
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool3d((last_duration, last_size, last_size), stride=1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if pretrained:
            print('Loading pretrained weights for squeezenet...')
            self.__load_pretrained_weights_squeezenet()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

    def __load_pretrained_weights_squeezenet(self):
        corresp_name = ['features.0.weight', 'features.0.bias', 'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.1.num_batches_tracked', 'features.4.squeeze.weight', 'features.4.squeeze.bias',
                        'features.4.squeeze_bn.weight', 'features.4.squeeze_bn.bias', 'features.4.squeeze_bn.running_mean', 'features.4.squeeze_bn.running_var', 'features.4.squeeze_bn.num_batches_tracked', 'features.4.expand1x1.weight', 'features.4.expand1x1.bias',
                        'features.4.expand1x1_bn.weight', 'features.4.expand1x1_bn.bias', 'features.4.expand1x1_bn.running_mean', 'features.4.expand1x1_bn.running_var', 'features.4.expand1x1_bn.num_batches_tracked', 'features.4.expand3x3.weight',
                        'features.4.expand3x3.bias', 'features.4.expand3x3_bn.weight', 'features.4.expand3x3_bn.bias', 'features.4.expand3x3_bn.running_mean', 'features.4.expand3x3_bn.running_var', 'features.4.expand3x3_bn.num_batches_tracked',
                        'features.5.squeeze.weight', 'features.5.squeeze.bias', 'features.5.squeeze_bn.weight', 'features.5.squeeze_bn.bias', 'features.5.squeeze_bn.running_mean', 'features.5.squeeze_bn.running_var', 'features.5.squeeze_bn.num_batches_tracked',
                        'features.5.expand1x1.weight', 'features.5.expand1x1.bias', 'features.5.expand1x1_bn.weight', 'features.5.expand1x1_bn.bias', 'features.5.expand1x1_bn.running_mean', 'features.5.expand1x1_bn.running_var',
                        'features.5.expand1x1_bn.num_batches_tracked', 'features.5.expand3x3.weight', 'features.5.expand3x3.bias', 'features.5.expand3x3_bn.weight', 'features.5.expand3x3_bn.bias', 'features.5.expand3x3_bn.running_mean',
                        'features.5.expand3x3_bn.running_var', 'features.5.expand3x3_bn.num_batches_tracked', 'features.7.squeeze.weight', 'features.7.squeeze.bias', 'features.7.squeeze_bn.weight', 'features.7.squeeze_bn.bias', 'features.7.squeeze_bn.running_mean',
                        'features.7.squeeze_bn.running_var', 'features.7.squeeze_bn.num_batches_tracked', 'features.7.expand1x1.weight', 'features.7.expand1x1.bias', 'features.7.expand1x1_bn.weight', 'features.7.expand1x1_bn.bias',
                        'features.7.expand1x1_bn.running_mean', 'features.7.expand1x1_bn.running_var', 'features.7.expand1x1_bn.num_batches_tracked', 'features.7.expand3x3.weight', 'features.7.expand3x3.bias', 'features.7.expand3x3_bn.weight',
                        'features.7.expand3x3_bn.bias', 'features.7.expand3x3_bn.running_mean', 'features.7.expand3x3_bn.running_var', 'features.7.expand3x3_bn.num_batches_tracked', 'features.8.squeeze.weight', 'features.8.squeeze.bias',
                        'features.8.squeeze_bn.weight', 'features.8.squeeze_bn.bias', 'features.8.squeeze_bn.running_mean', 'features.8.squeeze_bn.running_var', 'features.8.squeeze_bn.num_batches_tracked', 'features.8.expand1x1.weight',
                        'features.8.expand1x1.bias', 'features.8.expand1x1_bn.weight', 'features.8.expand1x1_bn.bias', 'features.8.expand1x1_bn.running_mean', 'features.8.expand1x1_bn.running_var', 'features.8.expand1x1_bn.num_batches_tracked',
                        'features.8.expand3x3.weight', 'features.8.expand3x3.bias', 'features.8.expand3x3_bn.weight', 'features.8.expand3x3_bn.bias', 'features.8.expand3x3_bn.running_mean', 'features.8.expand3x3_bn.running_var',
                        'features.8.expand3x3_bn.num_batches_tracked', 'features.10.squeeze.weight', 'features.10.squeeze.bias', 'features.10.squeeze_bn.weight', 'features.10.squeeze_bn.bias', 'features.10.squeeze_bn.running_mean',
                        'features.10.squeeze_bn.running_var', 'features.10.squeeze_bn.num_batches_tracked', 'features.10.expand1x1.weight', 'features.10.expand1x1.bias', 'features.10.expand1x1_bn.weight', 'features.10.expand1x1_bn.bias',
                        'features.10.expand1x1_bn.running_mean', 'features.10.expand1x1_bn.running_var', 'features.10.expand1x1_bn.num_batches_tracked', 'features.10.expand3x3.weight', 'features.10.expand3x3.bias', 'features.10.expand3x3_bn.weight',
                        'features.10.expand3x3_bn.bias', 'features.10.expand3x3_bn.running_mean', 'features.10.expand3x3_bn.running_var', 'features.10.expand3x3_bn.num_batches_tracked', 'features.11.squeeze.weight', 'features.11.squeeze.bias',
                        'features.11.squeeze_bn.weight', 'features.11.squeeze_bn.bias', 'features.11.squeeze_bn.running_mean', 'features.11.squeeze_bn.running_var', 'features.11.squeeze_bn.num_batches_tracked', 'features.11.expand1x1.weight',
                        'features.11.expand1x1.bias', 'features.11.expand1x1_bn.weight', 'features.11.expand1x1_bn.bias', 'features.11.expand1x1_bn.running_mean', 'features.11.expand1x1_bn.running_var', 'features.11.expand1x1_bn.num_batches_tracked',
                        'features.11.expand3x3.weight', 'features.11.expand3x3.bias', 'features.11.expand3x3_bn.weight', 'features.11.expand3x3_bn.bias', 'features.11.expand3x3_bn.running_mean', 'features.11.expand3x3_bn.running_var',
                        'features.11.expand3x3_bn.num_batches_tracked', 'features.13.squeeze.weight', 'features.13.squeeze.bias', 'features.13.squeeze_bn.weight', 'features.13.squeeze_bn.bias', 'features.13.squeeze_bn.running_mean',
                        'features.13.squeeze_bn.running_var', 'features.13.squeeze_bn.num_batches_tracked', 'features.13.expand1x1.weight', 'features.13.expand1x1.bias', 'features.13.expand1x1_bn.weight', 'features.13.expand1x1_bn.bias',
                        'features.13.expand1x1_bn.running_mean', 'features.13.expand1x1_bn.running_var', 'features.13.expand1x1_bn.num_batches_tracked', 'features.13.expand3x3.weight', 'features.13.expand3x3.bias', 'features.13.expand3x3_bn.weight',
                        'features.13.expand3x3_bn.bias', 'features.13.expand3x3_bn.running_mean', 'features.13.expand3x3_bn.running_var', 'features.13.expand3x3_bn.num_batches_tracked', 'features.14.squeeze.weight', 'features.14.squeeze.bias',
                        'features.14.squeeze_bn.weight', 'features.14.squeeze_bn.bias', 'features.14.squeeze_bn.running_mean', 'features.14.squeeze_bn.running_var', 'features.14.squeeze_bn.num_batches_tracked', 'features.14.expand1x1.weight',
                        'features.14.expand1x1.bias', 'features.14.expand1x1_bn.weight', 'features.14.expand1x1_bn.bias', 'features.14.expand1x1_bn.running_mean', 'features.14.expand1x1_bn.running_var', 'features.14.expand1x1_bn.num_batches_tracked',
                        'features.14.expand3x3.weight', 'features.14.expand3x3.bias', 'features.14.expand3x3_bn.weight', 'features.14.expand3x3_bn.bias', 'features.14.expand3x3_bn.running_mean', 'features.14.expand3x3_bn.running_var',
                        'features.14.expand3x3_bn.num_batches_tracked']
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
    model = SqueezeNet3D(version=1.1, sample_size=112, sample_duration=16, num_classes=101, pretrained='pretrained/pretrained_squeezenet3D/kinetics_squeezenet_RGB_16_best.pth')
    # print(model)
    output = model(X) # [1, 101]
    print(output.shape)