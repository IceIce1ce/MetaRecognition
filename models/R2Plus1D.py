# https://github.com/jfzhang95/pytorch-video-recognition/blob/master/network/R2Plus1D_model.py
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch

class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if first_conv:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size = kernel_size
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = padding
            temporal_kernel_size = (3, 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (1, 0, 0)
            # from the official code, first conv's intermed_channels = 45
            intermed_channels = 45
            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size, stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)
            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()
        else:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = (0, padding[1], padding[2])
            temporal_kernel_size = (kernel_size[0], 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (padding[0], 0, 0)
            # compute the number of intermediary channels (M) using formula
            # from the paper section 3.5
            intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                                (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size, stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)
            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x

class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.
        # no pooling layers are used inside ResNet
        self.downsample = downsample
        # to allow for SAME padding
        padding = kernel_size // 2
        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))
        return self.relu(x + res)

class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock, downsample=False):
        super(SpatioTemporalResLayer, self).__init__()
        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)
        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x

class R2Plus1DNet(nn.Module):
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()
        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.conv1 = SpatioTemporalConv(3, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), first_conv=True)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x.view(-1, 512)

class R2Plus1D(nn.Module):
    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R2Plus1D, self).__init__()
        self.res2plus1d = R2Plus1DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)
        self.__init_weight()
        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res2plus1d(x)
        logits = self.linear(x)
        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def R2Plus1D_18(**kwargs):
    return R2Plus1D(layer_sizes=(2, 2, 2, 2), pretrained=False, **kwargs)

def R2Plus1D_34(**kwargs):
    return R2Plus1D(layer_sizes=(3, 4, 6, 3), pretrained=False, **kwargs)

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    model = R2Plus1D_18(num_classes=101)
    print(model)
    outputs = model(inputs) # [1, 101]
    print(outputs.shape)