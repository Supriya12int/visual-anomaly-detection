import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # load pretrained resnet18
        resnet = models.resnet18(pretrained=True)

        # remove classifier (FC layer)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # freeze weights (IMPORTANT for deployment)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Input : [B, 3, H, W]
        Output: [B, C, H', W']  (feature map)
        """
        return self.feature_extractor(x)
if __name__ == "__main__":
    model = ResNetFeatureExtractor()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(y.shape)
