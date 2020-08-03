import torch
import torch.nn as nn


class MissFirstModel(nn.Module):

    def __init__(self):
        super(MissFirstModel, self).__init__()
        self.name = "MissFirstModel"

        # linear
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)

        # conv
        # 4*4
        self.cv1 = nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1)
        # 8*8
        self.cv2 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1)
        self.cv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        # 16*16
        self.cv4 = nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1)
        self.cv5 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1)
        # 32*32
        self.cv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.cv7 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.cv8 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()  # 勾配消失問題に関わるから不適切

    def forward(self, E, G):
        x = torch.cat((E, G), dim=2)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = torch.reshape(x, (-1, 1, 4, 4))
        # 4*4
        x = self.act(self.cv1(x))
        x = self.up(x)
        # 8*8
        x = self.act(self.cv2(x))
        x = self.act(self.cv3(x))
        x = self.up(x)
        # 16*16
        x = self.act(self.cv4(x))
        x = self.act(self.cv5(x))
        x = self.up(x)
        # 32*32
        x = self.act(self.cv6(x))
        x = self.act(self.cv7(x))
        x = self.sig(self.cv8(x))

        return x


class FirstModel(nn.Module):

    def __init__(self):
        super(FirstModel, self).__init__()
        self.name = "FirstModel"

        # linear
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 16)
        self.fc4 = nn.Linear(16, 16)

        # conv
        # 4*4
        self.cv1 = nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1)
        # 8*8
        self.cv2 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1)
        self.cv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        # 16*16
        self.cv4 = nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1)
        self.cv5 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1)
        # 32*32
        self.cv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.cv7 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.cv8 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()  # 勾配消失問題に関わるから不適切

    def forward(self, E, G):
        x = torch.cat((E, G), dim=2)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = torch.reshape(x, (-1, 1, 4, 4))
        # 4*4
        x = self.act(self.cv1(x))
        x = self.up(x)
        # 8*8
        x = self.act(self.cv2(x))

        x = self.act(self.cv3(x))
        x = self.up(x)
        # 16*16
        x = self.act(self.cv4(x))
        x = self.act(self.cv5(x))
        x = self.up(x)
        # 32*32
        x = self.act(self.cv6(x))
        x = self.act(self.cv7(x))
        x = self.sig(self.cv8(x))

        return x


class FirstModelResNet(FirstModel):

    def __init__(self):
        super(FirstModelResNet, self).__init__()
        self.name = "FirstModelResNet"

    def forward(self, E, G):
        x = torch.cat((E, G), dim=2)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = torch.reshape(x, (-1, 1, 4, 4))
        # 4*4
        x = self.act(self.cv1(x))
        x = self.up(x)
        # 8*8
        x = self.act(self.cv2(x))
        h = x
        x = self.act(self.cv3(x))
        x = x+h
        x = self.up(x)
        # 16*16
        x = self.act(self.cv4(x))
        h = x
        x = self.act(self.cv5(x))
        x = x+h
        x = self.up(x)
        # 32*32
        x = self.act(self.cv6(x))
        x = self.act(self.cv7(x))
        x = self.sig(self.cv8(x))

        return x


class FirstModel_252(FirstModel):

    def __init__(self):
        super(FirstModel_252, self).__init__()
        self.name = "FirstModel_252"
        # conv
        # 4*4
        self.cv1 = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1)
        # 8*8
        self.cv2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, E, G):
        x = torch.cat((E, G), dim=2)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = torch.reshape(x, (-1, 1, 4, 4))
        # 4*4
        x = self.act(self.cv1(x))
        x = self.up(x)
        # 8*8
        x = self.act(self.cv2(x))
        x = self.act(self.cv3(x))
        x = self.up(x)
        # 16*16
        x = self.act(self.cv4(x))
        x = self.act(self.cv5(x))
        x = self.up(x)
        # 32*32
        x = self.act(self.cv6(x))
        x = self.act(self.cv7(x))
        x = self.sig(self.cv8(x))

        return x


class FirstModelBatch(FirstModel):

    def __init__(self):
        super(FirstModelBatch, self).__init__()
        self.name = "FirstModelBatch"
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.cvbn1 = nn.BatchNorm2d(512)
        self.cvbn2 = nn.BatchNorm2d(256)
        self.cvbn3 = nn.BatchNorm2d(128)
        self.cvbn4 = nn.BatchNorm2d(64)

    def forward(self, E, G):
        x = torch.cat((E, G), dim=2)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.act(self.bn3(self.fc3(x)))
        x = self.act(self.bn4(self.fc4(x)))
        x = torch.reshape(x, (-1, 1, 4, 4))
        # 4*4
        x = self.act(self.cvbn1(self.cv1(x)))
        x = self.up(x)
        # 8*8
        x = self.act(self.cvbn2(self.cv2(x)))
        x = self.act(self.cv3(x))
        x = self.up(x)
        # 16*16
        x = self.act(self.cvbn3(self.cv4(x)))
        x = self.act(self.cv5(x))
        x = self.up(x)
        # 32*32
        x = self.act(self.cvbn4(self.cv6(x)))
        x = self.act(self.cv7(x))
        x = self.sig(self.cv8(x))

        return x
