import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, seg_label):
        super(Generator, self).__init__()
        self.seg_label = seg_label
        # encode

        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.a2 = nn.ReLU()
        self.m1 = nn.MaxPool2d(2, 2)
        # 16*16
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.a3 = nn.ReLU()
        self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.a4 = nn.ReLU()
        self.m2 = nn.MaxPool2d(2, 2)
        # 8*8

        self.c7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.a7 = nn.ReLU()
        self.c8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.a8 = nn.ReLU()
        self.m4 = nn.MaxPool2d(2, 2)
        # 4*4
        self.c9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.a9 = nn.ReLU()

        # decode

        self.b1 = nn.Conv2d(513, 512, kernel_size=3,
                            stride=1, padding=1)  # >512*4*4

        self.conv_img1 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1)
        self.conv_img3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)

        self.conv_img4 = nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1)
        self.conv_img5 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1)

        self.conv_img6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv_img7 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.conv_img8 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        self.rel = nn.ReLU()
        self.bn = nn.Sigmoid()  # 勾配消失問題に関わるから不適切

        # initialize_weights(self)

    def forward(self, bf, vol, seg):
        x = torch.cat((bf, seg), dim=1)
        x = self.c1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.a2(x)
        x = self.m1(x)
        x = self.c3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.a4(x)
        x = self.m2(x)
        x = self.c7(x)
        x = self.a7(x)
        x = self.c8(x)
        x = self.a8(x)
        x = self.m4(x)
        x = self.c9(x)
        x = self.a9(x)

        # concat vol
        x2 = vol
        x = torch.cat((x, x2), dim=1)

        x = self.b1(x)
        x = self.up(x)
        x = self.conv_img1(x)
        x = self.rel(x)
        x = self.conv_img3(x)
        x = self.rel(x)

        x = self.up(x)
        x = self.conv_img4(x)
        x = self.rel(x)
        x = self.conv_img5(x)
        x = self.rel(x)

        x = self.up(x)
        x = self.conv_img6(x)
        x = self.rel(x)
        x = self.conv_img7(x)
        x = self.rel(x)

        x = self.conv_img8(x)
        x = self.bn(x)

        return x
