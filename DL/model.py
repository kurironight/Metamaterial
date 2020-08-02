import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # encode

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        # 4*4
        self.c3 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        # 32*32
        self.c4 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        self.up = nn.Upsample(scale_factor=8)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()  # 勾配消失問題に関わるから不適切

        # initialize_weights(self)

    def forward(self, E, G):
        x = torch.cat((E, G), dim=2)
        x = self.rel(self.fc1(x))
        x = self.rel(self.fc2(x))
        x = torch.reshape(x, (-1, 1, 4, 4))
        # 4*4
        x = self.rel(self.c3(x))
        x = self.up(x)
        # 32*32
        x = self.c4(x)
        x = self.sig(x)

        return x
