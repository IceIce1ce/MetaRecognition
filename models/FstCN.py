# https://github.com/MRzzm/action-recognition-models-pytorch/blob/master/3DCNN/FstCN/FstCN.py
import torch
import torch.nn as nn
from torch.nn import init

class TCL(nn.Module):
    def __init__(self, in_channels,init_weights):
        super(TCL, self).__init__()
        # Conv(32, 3, 1)
        self.branch1 = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                                     nn.ReLU(True),
                                     nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                                     nn.Dropout(0.5))
        # Conv(32, 5, 1)
        self.branch2 = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0)),
                                     nn.ReLU(True),
                                     nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                                     nn.Dropout(0.5))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        res1 = self.branch1(x)
        res2 = self.branch2(x)
        return torch.cat([res1, res2], 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Conv3d):
                        init.xavier_uniform_(n.weight)
                        init.constant_(n.bias, 0)

# input_size: 16x204x204
class FstCN(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(FstCN, self).__init__()
        # Conv(96, 7, 2), Pooling(3, 2), norm is optional
        self.SCL1 = nn.Sequential(nn.Conv3d(3, 96, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
                                  nn.ReLU(True),
                                  nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2),
                                  nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)))
        # Conv(256, 5, 2), Pooling(3, 2), norm is optional
        self.SCL2 = nn.Sequential(nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
                                  nn.ReLU(True),
                                  nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2),
                                  nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)))
        # Conv(512, 3, 1)
        self.SCL3 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                                  nn.ReLU(True))
        # Conv(512, 3, 1)
        self.SCL4 = nn.Sequential(nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                                  nn.ReLU(True))
        # Paper: Conv3d -> SCL
        self.Parallel_temporal = nn.Sequential(nn.Conv3d(512, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                                               nn.MaxPool3d((1, 3, 3), stride=(1, 3, 3)),
                                               TCL(in_channels=128, init_weights=init_weights))
        # Paper: Conv2d -> SCL
        self.Parallel_spatial = nn.Sequential(nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                              nn.MaxPool2d((3, 3), stride=(3, 3)))
        self.tem_fc = nn.Sequential(nn.Linear(768, 4096), # change to (8192, 4096) when input size is 204 x 204
                                    nn.Dropout(),
                                    nn.Linear(4096, 2048))
        self.spa_fc = nn.Sequential(nn.Linear(512, 4096), # change to (2048, 4096) when input size is 204 x 204
                                    nn.Dropout(),
                                    nn.Linear(4096, 2048))
        self.fc = nn.Linear(4096, 2048)
        self.out = nn.Linear(2048, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, clip):
        video_idx = torch.arange(7) # [0, 1, 2, 3, 4, 5, 6]
        offset = video_idx + 9 # [9, 10, 11, 12, 13, 14, 15], dt is 9, default dt of paper is 5
        clip_diff = clip[:, :, video_idx] - clip[:, :, offset] # [32, 3, 7, 112, 112]
        clip = clip[:, :, video_idx] # [32, 3, 7, 112, 112], should try to use all 16 frames instead of video_idx
        # clip_diff = clip[:, :, 9:, :, :] # 7 frames
        # clip = clip[:, :, 0:9, :, :] # 9 frames
        clip_all = torch.cat([clip, clip_diff], 2) # [32, 3, 14, 112, 112]
        clip_len = clip.size(2) # 9 frames
        # Clip -> Spatial Conv Layer
        clip_all = self.SCL1(clip_all)
        clip_all = self.SCL2(clip_all)
        clip_all = self.SCL3(clip_all)
        clip_all = self.SCL4(clip_all)
        clip = clip_all[:, :, :clip_len, :, :]
        clip_diff = clip_all[:, :, clip_len:, :, :]
        clip = torch.squeeze(clip[:, :, clip.size(2)//2, :, :]) # [32, 512, 6, 6]
        # second branch of parallel conv: SCL -> FC -> FC
        clip = self.Parallel_spatial(clip) # [32, 128, 2, 2]
        clip = self.spa_fc(clip.view(clip.size(0), -1)) # [32, 512] -> [32, 2048]
        # first brnahc of parallel conv: SCL -> transformation -> TCL -> FC -> FC, TP operator is optional
        clip_diff = self.Parallel_temporal(clip_diff) # [32, 512, 7, 6, 6] -> [32, 64, 3, 2, 2]
        clip_diff = self.tem_fc(clip_diff.view(clip_diff.size(0), -1)) # [32, 768] -> [32, 2048]
        res = torch.cat([clip, clip_diff], 1) # [32, 4096]
        res = self.fc(res)
        res = self.out(res)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Conv3d):
                        init.xavier_uniform_(n.weight)
                        if n.bias is not None:
                            init.constant_(n.bias, 0)
                    elif isinstance(n, nn.Conv2d):
                        init.xavier_uniform_(n.weight)
                        if n.bias is not None:
                            init.constant_(n.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112) # Paper: (1, 3, 16, 204, 204), use 204 instead of 224 to save memory
    model = FstCN(num_classes=101)
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)