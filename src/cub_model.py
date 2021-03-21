import torch.nn as nn
from googlenet_pytorch import GoogLeNet 

class GoogLeNetExtension(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = GoogLeNet.from_pretrained('googlenet')
        self.pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    
    def forward(self, x):
        x = self.pretrained.extract_features(x)
        x = self.pool_layer(x)
        return x    
