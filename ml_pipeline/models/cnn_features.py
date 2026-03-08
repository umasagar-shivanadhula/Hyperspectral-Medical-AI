
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSI_CNN(nn.Module):

    def __init__(self, bands):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, (3,3,7))
        self.conv2 = nn.Conv3d(16, 32, (3,3,5))

        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(32, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
