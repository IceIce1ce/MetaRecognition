import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet3D(nn.Module):
    def __init__(self, num_classes=101):
        super(Lenet3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 6, kernel_size=(5, 5, 5))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    X = torch.rand(1, 3, 16, 112, 112)
    model = Lenet3D(num_classes=101)
    print(model)
    output = model(X) # [1, 101]
    print(output.shape)