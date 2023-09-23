import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 161 * 161 =pool=> 80 * 80 =upsample=> 160 * 160
        # => TF.resize로 아래에서 해결함.

        # Down Part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up Part of U-Net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        out = x
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)
        skip_connections = skip_connections[::-1]  # reverse

        for idx in range(0, len(self.ups), 2):
            out = self.ups[idx](out)  # ConvTransposed
            skip_connection = skip_connections[idx // 2]

            # assert out.shape == skip_connection.shape
            if out.shape != skip_connection.shape:
                out = TF.resize(out, size=skip_connection.shape[2:])  # [2:] : H*W dimension

            concat_skip = torch.cat((skip_connection, out), dim=1)  # dim=1 : channel dim
            out = self.ups[idx + 1](concat_skip)

        return self.final_conv(out)


def test():
    model = UNET(in_channels=1, out_channels=1)
    x = torch.randn((3, 1, 160, 160))
    preds = model(x)

    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == '__main__':
    test()
