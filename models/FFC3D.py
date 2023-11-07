# https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
import torch
import torch.nn as nn

class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv1 = nn.Conv3d(channels, channels//r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv3d(channels//r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv3d(channels//r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x
        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))
        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv3d(in_channels=in_channels*2, out_channels=out_channels*2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm3d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, _, h, w = x.size() # [1, 3, 16, 224, 224]
        ffted = torch.fft.rfftn(x, s=(h, w), dim=(3, 4), norm='ortho') # [bs, c, h, w/2+1]: [1, 3, 224, 113]
        ffted = torch.cat([ffted.real, ffted.imag], dim=1) # [1, 6, 224, 113]
        ffted = self.conv_layer(ffted) # [bs, c*2, h, w/2+1]: [1, 64, 224, 113]
        ffted = self.relu(self.bn(ffted)) # [1, 64, 224, 113]
        ffted = torch.tensor_split(ffted, 2, dim=1) # [bs, c, h, w/2+1] -> Tuple([1, 32, 224, 113], [1, 32, 224, 113])
        ffted = torch.complex(ffted[0], ffted[1]) # [1, 32, 224, 113]
        output = torch.fft.irfftn(ffted, s=(h, w), dim=(3, 4), norm='ortho') # [1, 32, 224, 224]
        return output

class SpectralTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransformer, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, out_channels//2, kernel_size=1, groups=groups, bias=False),
                                   nn.BatchNorm3d(out_channels // 2),
                                   nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels//2, out_channels//2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels//2, out_channels//2, groups)
        self.conv2 = torch.nn.Conv3d(out_channels//2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x) # channel reduction, stride = 1: [1, 3, 16, 224, 224], stride = 2: [1, 3, 16, 112, 112]
        x = self.conv1(x) # [1, 16, 224, 224]
        output = self.fu(x) # fourier unit: [1, 16, 16, 224, 224]
        # local fourier unit
        if self.enable_lfu:
            _, c, _, h, w = x.shape # [1, 16, 16, 224, 224]
            split_no = 2
            split_s_h = h // split_no # 112
            split_s_w = w // split_no # 112
            # x: [1, 4, 16, 224, 224] -> split 112 -> [1, 4, 16, 112, 224] -> cat -> [1, 8, 16, 112, 224]
            xs = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            # [1, 8, 16, 112, 224] -> [1, 8, 16, 112, 112] -> [1, 16, 16, 112, 112]
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs) # fourier unit: [1, 16, 112, 112] --- err: [1, 16, 112, 112, 112]
            xs = xs.repeat(1, 1, 1, split_no, split_no).contiguous() # spatial shift: [1, 16, 224, 224]
        else:
            xs = 0
        output = self.conv2(x + output + xs) # channel promotion --- [1, 32, 16, 224, 224]
        return output

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride # 1
        in_cg = int(in_channels * ratio_gin) # input channel global: 0
        in_cl = in_channels - in_cg # input channel local: 3
        out_cg = int(out_channels * ratio_gout) # output channel global: 16
        out_cl = out_channels - out_cg # output channel local: 16
        self.ratio_gin = ratio_gin # 0
        self.ratio_gout = ratio_gout # 0.5
        # Section 3.1
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv3d
        self.l2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias) # (3, 16)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv3d
        self.l2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias) # (3, 16)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv3d
        self.g2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias) # (0, 16)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransformer
        self.g2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu) # (0, 16)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0) # local part: [1, 3, 224, 224], global part: 0
        y_l, y_g = 0, 0 # output local and output global
        if self.ratio_gout != 1:
            y_l = self.l2l(x_l) + self.g2l(x_g) # Eq 1
        if self.ratio_gout != 0:
            y_g = self.g2g(x_g) + self.l2g(x_l) # Eq 2
        return y_l, y_g

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm3d, activation_layer=nn.Identity, enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, enable_lfu)
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
    X = torch.randn(1, 3, 16, 112, 112)
    # X = X if type(X) is tuple else (X, 0)
    conv1 = FFC_BN_ACT(3, 64, kernel_size=3, padding=1, ratio_gin=0, ratio_gout=0.5, stride=1, enable_lfu=True)
    conv2 = FFC_BN_ACT(64, 128, kernel_size=3, padding=1, ratio_gin=0.5, ratio_gout=0.5, stride=1, enable_lfu=True)
    conv3 = FFC_BN_ACT(128, 256, kernel_size=3, padding=1, ratio_gin=0.5, ratio_gout=0.5, stride=1, enable_lfu=True)
    conv4 = FFC_BN_ACT(256, 512, kernel_size=3, padding=1, ratio_gin=0.5, ratio_gout=0, stride=1, enable_lfu=True)
    X = conv1(X)
    X = conv2(X)
    X = conv3(X)
    pred = conv4(X)
    print(pred[0].shape)
    print(pred[1])