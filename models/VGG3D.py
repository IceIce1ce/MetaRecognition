import torch
from torch import nn

vgg_cfgs = {"vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M1"],
            "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M1"],
            "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M1"],
            "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M1"]}

def _make_layers(vgg_cfg):
    layers = []
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers += [nn.MaxPool3d((2, 2, 2), (2, 2, 2))]
        elif v == "M1":
            layers += [nn.MaxPool3d((1, 1, 1), (1, 1, 1))]
        else:
            conv3d = nn.Conv3d(in_channels, v, (3, 3, 3), (1, 1, 1), (1, 1, 1))
            layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, vgg_cfg, num_classes=101):
        super(VGG, self).__init__()
        self.features = _make_layers(vgg_cfg)
        self.avgpool = nn.AvgPool3d(kernel_size=1, stride=1) # need test without using AvgPool
        # self.classifier = nn.Linear(512 * 7 * 7, num_classes)
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(4096, 4096),
                                        nn.ReLU(True), nn.Dropout(0.5), nn.Linear(4096, num_classes)) # same as C3D
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

def VGG3D_11(**kwargs):
    return VGG(vgg_cfgs["vgg11"], **kwargs)

def VGG3D_13(**kwargs):
    return VGG(vgg_cfgs["vgg13"], **kwargs)

def VGG3D_16(**kwargs):
    return VGG(vgg_cfgs["vgg16"], **kwargs)

def VGG3D_19(**kwargs):
    return VGG(vgg_cfgs["vgg19"], **kwargs)

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112) # use [1, 3, 32, 224, 224] if replace M1 = M
    model = VGG3D_19(num_classes=101)
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)