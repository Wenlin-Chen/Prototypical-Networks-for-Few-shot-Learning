import torch.nn as nn


class EmbeddingNet(nn.Module):

    def __init__(self, img_channels, hidden_channels, embedded_channels):
        super(EmbeddingNet, self).__init__()
        self.embedder = nn.Sequential(
            self.conv_block(img_channels, hidden_channels),
            self.conv_block(hidden_channels, hidden_channels),
            self.conv_block(hidden_channels, hidden_channels),
            self.conv_block(hidden_channels, embedded_channels)
        )

    def conv_block(self, in_channels, out_channels, conv_kernel=3, conv_stride=1, conv_padding=1, pool_kernel=2):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel)
        )
        return block

    def forward(self, x):
        x = self.embedder(x)
        return x.view(x.size(0), -1)


