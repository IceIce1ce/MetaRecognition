# https://github.com/AlexHex7/Non-local_pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, mode='embedded_gaussian'):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['gaussian', 'embedded_gaussian', 'dot_product', 'concatenation']
        self.mode = mode
        self.dimension = dimension # 1
        self.sub_sample = sub_sample
        self.in_channels = in_channels # 512
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2 # 256
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0) # (512, 256, 1, 1, 0)
        if bn_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), bn(self.in_channels))
            # section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            # section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        if self.mode == "embedded_gaussian" or self.mode == "dot_product" or self.mode == "concatenation":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0) # (512, 256, 1, 1, 0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0) # (512, 256, 1, 1, 0)
        if self.mode == "concatenation":
            self.W_f = nn.Sequential(nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU())
        if sub_sample:
            if self.mode == "embedded_gaussian" or self.mode == "dot_product" or self.mode == "concatenation":
                self.g = nn.Sequential(self.g, max_pool_layer)
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            elif self.mode == 'gaussian':
                self.g = nn.Sequential(self.g, max_pool_layer)
                self.phi = max_pool_layer

    def forward(self, x):
        # x -> [640, 512, 32]
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # [640, 256, 32]
        g_x = g_x.permute(0, 2, 1) # [640, 32, 256]
        if self.mode == 'gaussian':
            if self.sub_sample:
                phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
            else:
                phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode == "embedded_gaussian" or self.mode == "dot_product":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # [640, 256, 32]
            theta_x = theta_x.permute(0, 2, 1) # [640, 32, 256]
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [640, 256, 32]
            f = torch.matmul(theta_x, phi_x) # [640, 32, 32]
        elif self.mode == "concatenation":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        if self.mode == "gaussian" or self.mode == "embedded_gaussian":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot_product" or self.mode == "concatenation":
            N = f.size(-1) # number of position in x, 32
            f_div_C = f / N # [640, 32, 32]
        y = torch.matmul(f_div_C, g_x) # [640, 32, 256]
        y = y.permute(0, 2, 1).contiguous() # [640, 256, 32]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # [640, 256, 32]
        W_y = self.W(y) # [640, 512, 32]
        z = W_y + x # residual connection, [640, 512, 32]
        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, mode='embedded_gaussian'):
        super(NONLocalBlock1D, self).__init__(in_channels, inter_channels=inter_channels, dimension=1, sub_sample=sub_sample, bn_layer=bn_layer, mode=mode)

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, mode='embedded_gaussian'):
        super(NONLocalBlock2D, self).__init__(in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer, mode=mode)

class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, mode='embedded_gaussian'):
        super(NONLocalBlock3D, self).__init__(in_channels, inter_channels=inter_channels, dimension=3, sub_sample=sub_sample, bn_layer=bn_layer, mode=mode)

if __name__ == '__main__':
    for (sub_sample_, bn_layer_) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.randn(1, 3, 20)
        net = NONLocalBlock1D(3, sub_sample=sub_sample_, bn_layer=bn_layer_, mode='gaussian')
        out = net(img)
        print(out.shape)

        img = torch.randn(1, 3, 20, 20)
        net = NONLocalBlock2D(3, sub_sample=sub_sample_, bn_layer=bn_layer_, mode='gaussian')
        out = net(img)
        print(out.shape)

        img = torch.randn(1, 3, 8, 20, 20)
        net = NONLocalBlock3D(3, sub_sample=sub_sample_, bn_layer=bn_layer_, mode='gaussian')
        out = net(img)
        print(out.shape)