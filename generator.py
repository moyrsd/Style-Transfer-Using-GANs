
import torch
import torch.nn as nn


class resblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_planes, out_planes,
                               kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_planes)
        self.batch_norm2 = nn.BatchNorm2d(out_planes)
        self.identity = nn.Sequential()

    def forward(self, input):
        identity = self.identity(input)
        input = self.conv1(input)
        input = self.batch_norm1(input)
        input = nn.ReLU()(input)
        input = self.conv2(input)
        input = self.batch_norm2(input)
        input = input + identity
        output = nn.ReLU()(input)
        return output


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride,
                      padding, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)


class transpose_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1, output_padding=1,
                               stride=stride, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)


class generator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        layers = []
        layers.append(conv_block(in_channels, 64, 1, 7, 3))
        layers.append(conv_block(64, 128, 2, 3, 1))
        layers.append(conv_block(128, 256, 2, 3, 1))
        for i in range(9):
            layers.append(resblock(256, 256))
        layers.append(transpose_conv_block(256, 128, 2))
        layers.append(transpose_conv_block(128, 64, 2))
        layers.append(conv_block(64, 3, 1, 7, 3))
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


def test():
    x = torch.randn((6, 3, 256, 256))
    model = generator(in_channels=3)
    pred = model(x)
    print(pred.shape)


if __name__ == "__main__":
    test()
