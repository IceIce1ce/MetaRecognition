# https://github.com/BizhuWu/LRCN_PyTorch/blob/main/model.py
import torch.nn as nn
import torch
import torchvision.models as models

class LRCN(nn.Module):
    def __init__(self, bidirectional=True, num_classes=101):
        super(LRCN, self).__init__()
        self.featureExtractor = models.alexnet(pretrained=True)
        self.featureExtractor.classifier = nn.Sequential(*list(self.featureExtractor.classifier.children())[:-5]) # remove 2 last FC layers
        # self.featureExtractor = models.resnet152(pretrained=True)
        # for param in self.featureExtractor.parameters():
        #     param.requires_grad = False
        # self.featureExtractor.fc = nn.Linear(self.featureExtractor.fc.in_features, 512)
        self.lstm = nn.LSTM(input_size=4096, hidden_size=256, num_layers=2, batch_first=True, bidirectional=bidirectional) # [batch, seq, feature]
        self.out = nn.Linear(2*256 if bidirectional else 256, num_classes)

    def forward(self, video_clip):
        fea = torch.empty(size=(video_clip.size()[0], video_clip.size()[2], 4096)).to(video_clip.device) # [1, 16, 4096]
        for t in range(0, video_clip.size()[2]):
            frame = video_clip[:, :, t, :, :]
            frame_feature = self.featureExtractor(frame)
            fea[:, t, :] = frame_feature
        x, _ = self.lstm(fea)
        x = self.out(x)
        x = torch.mean(x, dim=1)
        return x

if __name__ == '__main__':
    X = torch.rand(1, 3, 16, 112, 112)
    model = LRCN(num_classes=101)
    print(model)
    output = model(X)
    print(output.size())