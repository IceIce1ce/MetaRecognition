# https://github.com/Tushar-N/pytorch-resnet3d/blob/master/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        outplanes = planes * 4
        self.nl = NonLocalBlock(outplanes, outplanes, outplanes//2) if use_nl else None

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
        if self.nl is not None:
            out = self.nl(out)
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_inner = dim_inner  
        self.dim_out = dim_out
        self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn = nn.BatchNorm3d(dim_out)

    def forward(self, x):
        residual = x
        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)
        theta_shape_5d = theta.shape
        theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(batch_size, self.dim_inner, -1)
        theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner**-.5)
        p = F.softmax(theta_phi_sc, dim=-1)
        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)
        out = self.out(t)
        out = self.bn(out)
        out = out + residual
        return out

class I3Res50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False, pretrained=None):
        super(I3Res50, self).__init__()
        self.pretrained = pretrained
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.5)
        self.__init_weight()
        if pretrained:
            if use_nl == True:
                print('Loading pretrained weights for ResNet50_I3D non local...')
                self.__load_pretrained_weights_nl()
            else:
                print('Loading pretrained weights for ResNet50_I3D...')
                self.__load_pretrained_weights()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0]!=1:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                                       nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i], i % nonlocal_mod == nonlocal_mod - 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def __load_pretrained_weights(self):
        corresp_name = {'conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias',
                        'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean',
                        'layer1.0.bn2.running_var', 'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var',
                        'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias', 'layer1.0.downsample.1.running_mean',
                        'layer1.0.downsample.1.running_var', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean',
                        'layer1.1.bn1.running_var', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
                        'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.2.conv1.weight',
                        'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight',
                        'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias',
                        'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean',
                        'layer2.0.bn1.running_var', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
                        'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var', 'layer2.0.downsample.0.weight',
                        'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var',
                        'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.conv2.weight',
                        'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight',
                        'layer2.1.bn3.bias', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias',
                        'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean',
                        'layer2.2.bn2.running_var', 'layer2.2.conv3.weight', 'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var',
                        'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.conv2.weight',
                        'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.conv3.weight', 'layer2.3.bn3.weight',
                        'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias',
                        'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean',
                        'layer3.0.bn2.running_var', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var',
                        'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean',
                        'layer3.0.downsample.1.running_var', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean',
                        'layer3.1.bn1.running_var', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
                        'layer3.1.conv3.weight', 'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.2.conv1.weight',
                        'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight',
                        'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias',
                        'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean',
                        'layer3.3.bn1.running_var', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
                        'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.4.conv1.weight',
                        'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight',
                        'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.conv3.weight', 'layer3.4.bn3.weight', 'layer3.4.bn3.bias',
                        'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean',
                        'layer3.5.bn1.running_var', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var',
                        'layer3.5.conv3.weight', 'layer3.5.bn3.weight', 'layer3.5.bn3.bias', 'layer3.5.bn3.running_mean', 'layer3.5.bn3.running_var', 'layer4.0.conv1.weight',
                        'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight',
                        'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.conv3.weight', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias',
                        'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias',
                        'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias',
                        'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean',
                        'layer4.1.bn2.running_var', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var',
                        'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.conv2.weight',
                        'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight',
                        'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var'}
        p_dict = torch.load(self.pretrained)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def __load_pretrained_weights_nl(self):
        corresp_name = {'conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight',
                        'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias',
                        'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean',
                        'layer1.0.bn3.running_var', 'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias',
                        'layer1.0.downsample.1.running_mean', 'layer1.0.downsample.1.running_var', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
                        'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean',
                        'layer1.1.bn2.running_var', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var',
                        'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.conv2.weight',
                        'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight',
                        'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias',
                        'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean',
                        'layer2.0.bn2.running_var', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var',
                        'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean',
                        'layer2.0.downsample.1.running_var', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean',
                        'layer2.1.bn1.running_var', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
                        'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.nl.theta.weight',
                        'layer2.1.nl.theta.bias', 'layer2.1.nl.phi.weight', 'layer2.1.nl.phi.bias', 'layer2.1.nl.g.weight', 'layer2.1.nl.g.bias', 'layer2.1.nl.out.weight',
                        'layer2.1.nl.out.bias', 'layer2.1.nl.bn.weight', 'layer2.1.nl.bn.bias', 'layer2.1.nl.bn.running_mean', 'layer2.1.nl.bn.running_var',
                        'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.conv2.weight',
                        'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.conv3.weight', 'layer2.2.bn3.weight',
                        'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias',
                        'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean',
                        'layer2.3.bn2.running_var', 'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var',
                        'layer2.3.nl.theta.weight', 'layer2.3.nl.theta.bias', 'layer2.3.nl.phi.weight', 'layer2.3.nl.phi.bias', 'layer2.3.nl.g.weight', 'layer2.3.nl.g.bias',
                        'layer2.3.nl.out.weight', 'layer2.3.nl.out.bias', 'layer2.3.nl.bn.weight', 'layer2.3.nl.bn.bias', 'layer2.3.nl.bn.running_mean', 'layer2.3.nl.bn.running_var',
                        'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.conv2.weight',
                        'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight',
                        'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight',
                        'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight',
                        'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias',
                        'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.conv3.weight', 'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean',
                        'layer3.1.bn3.running_var', 'layer3.1.nl.theta.weight', 'layer3.1.nl.theta.bias', 'layer3.1.nl.phi.weight', 'layer3.1.nl.phi.bias', 'layer3.1.nl.g.weight',
                        'layer3.1.nl.g.bias', 'layer3.1.nl.out.weight', 'layer3.1.nl.out.bias', 'layer3.1.nl.bn.weight', 'layer3.1.nl.bn.bias', 'layer3.1.nl.bn.running_mean',
                        'layer3.1.nl.bn.running_var', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var',
                        'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.conv3.weight',
                        'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight',
                        'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias',
                        'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean',
                        'layer3.3.bn3.running_var', 'layer3.3.nl.theta.weight', 'layer3.3.nl.theta.bias', 'layer3.3.nl.phi.weight', 'layer3.3.nl.phi.bias', 'layer3.3.nl.g.weight',
                        'layer3.3.nl.g.bias', 'layer3.3.nl.out.weight', 'layer3.3.nl.out.bias', 'layer3.3.nl.bn.weight', 'layer3.3.nl.bn.bias', 'layer3.3.nl.bn.running_mean',
                        'layer3.3.nl.bn.running_var', 'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var',
                        'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.conv3.weight',
                        'layer3.4.bn3.weight', 'layer3.4.bn3.bias', 'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight',
                        'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias',
                        'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.conv3.weight', 'layer3.5.bn3.weight', 'layer3.5.bn3.bias', 'layer3.5.bn3.running_mean',
                        'layer3.5.bn3.running_var', 'layer3.5.nl.theta.weight', 'layer3.5.nl.theta.bias', 'layer3.5.nl.phi.weight', 'layer3.5.nl.phi.bias', 'layer3.5.nl.g.weight',
                        'layer3.5.nl.g.bias', 'layer3.5.nl.out.weight', 'layer3.5.nl.out.bias', 'layer3.5.nl.bn.weight', 'layer3.5.nl.bn.bias', 'layer3.5.nl.bn.running_mean',
                        'layer3.5.nl.bn.running_var', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var',
                        'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.conv3.weight',
                        'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var', 'layer4.0.downsample.0.weight',
                        'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var',
                        'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.conv2.weight',
                        'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight',
                        'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias',
                        'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean',
                        'layer4.2.bn2.running_var', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var'}
        p_dict = torch.load(self.pretrained)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

def I3D_Resnet50(num_classes):
    net = I3Res50(num_classes=num_classes, use_nl=False, pretrained='pretrained/pretrained_resnet_i3d/i3d_r50_kinetics.pth')
    return net

def I3D_Resnet50_NL(num_classes):
    net = I3Res50(num_classes=num_classes, use_nl=True, pretrained='pretrained/pretrained_resnet_i3d/i3d_r50_nl_kinetics.pth')
    return net

if __name__=='__main__':
    X = torch.randn(1, 3, 16, 112, 112)
    model = I3D_Resnet50(num_classes=101)
    pred = model(X)
    print(pred.shape)
