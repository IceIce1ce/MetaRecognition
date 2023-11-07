# https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
import torch
import torch.nn as nn

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv1d(in_channels=in_channels*2, out_channels=out_channels*2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [640, 128, 32]
        batch, _, _ = x.size()
        r_size = x.size()
        ffted = torch.rfft(x, signal_ndim=1, normalized=True) # [640, 128, 17, 2]
        ffted = ffted.permute(0, 1, 3, 2).contiguous() # [640, 128, 2, 17]
        ffted = ffted.view((batch, -1,) + ffted.size()[3:]) # [64, 256, 17]
        ffted = self.conv_layer(ffted) # [640, 256, 17]
        ffted = self.relu(self.bn(ffted)) # [640, 256, 17]
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 2).contiguous() # [640, 128, 17, 2]
        output = torch.irfft(ffted, signal_ndim=1, signal_sizes=r_size[2:], normalized=True) # [640, 128, 32]
        return output

class SpectralTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(SpectralTransformer, self).__init__()
        if stride == 2:
            self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//2, kernel_size=1, groups=groups, bias=False),
                                   nn.BatchNorm1d(out_channels // 2),
                                   nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels//2, out_channels//2, groups)
        self.conv2 = torch.nn.Conv1d(out_channels//2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x) # [640, 512, 32]
        x = self.conv1(x) # [640, 128, 32]
        output = self.fu(x) # [640, 128, 32]
        output = self.conv2(x + output) # [640, 128, 32]
        return output

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        in_cg = int(in_channels * ratio_gin) # input channel global
        in_cl = in_channels - in_cg # input channel local
        out_cg = int(out_channels * ratio_gout) # output channel global
        out_cl = out_channels - out_cg # output channel local
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv1d
        self.l2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv1d
        self.l2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv1d
        self.g2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransformer
        self.g2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        y_l, y_g = 0, 0
        if self.ratio_gout != 1:
            y_l = self.l2l(x_l) + self.g2l(x_g)
        if self.ratio_gout != 0:
            y_g = self.g2g(x_g) + self.l2g(x_l)
        return y_l, y_g

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm1d, activation_layer=nn.Identity):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

if __name__ == '__main__':
    X = torch.randn(640, 2048, 32)
    conv1 = FFC_BN_ACT(2048, 1024, kernel_size=3, padding=1, ratio_gin=0, ratio_gout=0.5, stride=1)
    conv2 = FFC_BN_ACT(1024, 512, kernel_size=3, padding=1, ratio_gin=0.5, ratio_gout=0.5, stride=1)
    conv3 = FFC_BN_ACT(512, 512, kernel_size=3, padding=1, ratio_gin=0.5, ratio_gout=0, stride=1)
    X = conv1(X)
    X = conv2(X)
    pred = conv3(X)
    print(pred[0].shape)
    print(pred[1])