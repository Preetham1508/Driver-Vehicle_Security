import torch
from torchvision.models import resnet50
from torch import nn
from torch.nn import functional as F


class FaceNetModel(nn.Module):
    def __init__(self, emd_size=256):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.faceNet = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(2048, emd_size)
        self.l2_norm = F.normalize

    def forward(self, x):
        x = self.faceNet(x)
        x = self.fc(x)
        x = self.l2_norm(x, dim=1) * 10
        return x
    def forward_class(self, x):
        x = self.forward(x)
        x = self.fc_class(x)
        return x


if __name__ == '__main__':
    model = FaceNetModel()
    model(torch.ones(2, 3, 128, 128))
