# https://github.com/r1c7/SlowFastNetworks/blob/master/lib/slowfastnet.py
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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

class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=101, dropout=0.5):
        super(SlowFast, self).__init__()
        self.fast_inplanes = 8
        self.slow_inplanes = 64 + 64 // 8 * 2
        # Fast pathway
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(block, 64, layers[3], stride=2, head_conv=3)
        # Lateral connections
        self.lateral_p1 = nn.Conv3d(8, 16, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32, 64, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64, 128, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128, 256, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        # Slow pathway
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(block, 512, layers[3], stride=2, head_conv=3)
        # Classifier
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes + 2048, class_num, bias=False)

    def forward(self, input):
        # Fast pathway
        fast, lateral = self.FastPath(input[:, :, ::2, :, :]) # [32, 256]
        # Slow pathway, T = 16
        slow = self.SlowPath(input[:, :, ::16, :, :], lateral) # [32, 2048]
        x = torch.cat([slow, fast], dim=1) # [32, 2304]
        x = self.dp(x) # [32, 2304]
        x = self.fc(x) # [32, 101]
        return x

    def SlowPath(self, input, lateral):
        # input: [32, 3, 1, 112, 112], lateral[0]: [32, 16, 1, 28, 28], lateral[1]: [32, 64, 1, 28, 28]
        # lateral[2]: [32, 128, 1, 14, 14], lateral[3]: [32, 256, 1, 7, 7]
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x) # [32, 64, 1, 28, 28]
        x = torch.cat([x, lateral[0]], dim=1) # [32, 80, 1, 28, 28]
        x = self.slow_res2(x) # [32, 256, 1, 28, 28]
        x = torch.cat([x, lateral[1]], dim=1) # [32, 320, 1, 28, 28]
        x = self.slow_res3(x) # [32, 512, 1, 14, 14]
        x = torch.cat([x, lateral[2]], dim=1) # [32, 640, 1, 14, 14]
        x = self.slow_res4(x) # [32, 1024, 1, 7, 7]
        x = torch.cat([x, lateral[3]], dim=1) # [32, 1280, 1, 7, 7]
        x = self.slow_res5(x) # [32, 2048, 1, 4, 4]
        x = nn.AdaptiveAvgPool3d(1)(x) # [32, 2048, 1, 1, 1]
        x = x.view(-1, x.size(1)) # [32, 2048]
        return x

    def FastPath(self, input):
        # input: [32, 3, 8, 112, 112]
        lateral = []
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x) # [32, 8, 8, 28, 28]
        lateral_p = self.lateral_p1(pool1) # [32, 16, 1, 28, 28]
        lateral.append(lateral_p)
        res2 = self.fast_res2(pool1) # [32, 32, 8, 28, 28]
        lateral_res2 = self.lateral_res2(res2) # [32, 64, 1, 28, 28]
        lateral.append(lateral_res2)
        res3 = self.fast_res3(res2) # [32, 64, 8, 14, 14]
        lateral_res3 = self.lateral_res3(res3) # [32, 128, 1, 14, 14]
        lateral.append(lateral_res3)
        res4 = self.fast_res4(res3) # [32, 128, 8, 7, 7]
        lateral_res4 = self.lateral_res4(res4) # [32, 256, 1, 7, 7]
        lateral.append(lateral_res4)
        res5 = self.fast_res5(res4) # [32, 256, 8, 4, 4]
        x = nn.AdaptiveAvgPool3d(1)(res5) # [32, 256, 1, 1, 1]
        x = x.view(-1, x.size(1)) # [32, 256]
        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.fast_inplanes, planes * block.expansion, kernel_size=1, stride=(1, stride, stride), bias=False),
                                       nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.slow_inplanes, planes * block.expansion, kernel_size=1, stride=(1, stride, stride), bias=False),
                                       nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion + planes * block.expansion // 8 * 2
        return nn.Sequential(*layers)

def SlowFastResnet50(**kwargs):
    return SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)

def SlowFastResnet101(**kwargs):
    return SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)

def SlowFastResnet152(**kwargs):
    return SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)

def SlowFastResnet200(**kwargs):
    return SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = SlowFastResnet50(class_num=101)
    output = model(X) # [1, 101]
    print(output.shape)