
import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm):
        super().__init__()
        if (norm):
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride,
                          1, bias=True, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride,
                          1, bias=True, padding_mode="reflect"),
                nn.LeakyReLU(0.2)
            )

    def forward(self, input):
        return self.conv(input)


class discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        layers.append(conv_block(in_channels, features[0], 2, norm=False))
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(conv_block(in_channels, feature,
                          1 if feature == features[-1] else 2, norm=True))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4,
                      stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return torch.sigmoid(self.model(input))


def test():
    x = torch.randn((1, 3, 256, 256))
    model = discriminator(in_channels=3)
    pred = model(x)
    print(pred.shape)


if __name__ == "__main__":
    test()
