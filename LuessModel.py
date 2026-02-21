import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels = 1,convLayers=2):
        super().__init__()

        self.act = nn.LeakyReLU(inplace=True)
        layers = []
        

        for _ in range(convLayers-1):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(self.act)

        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(channels))

        self.net = nn.Sequential(
            *layers
        )


    def forward(self,x):
        res = x
        out = self.net(x)
        out += res
        return self.act(out)


class LuessModel(nn.Module):
    def __init__(self, num_filters=64, num_res_blocks = 8):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(19,num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU()
        )

        self.res_tower = nn.Sequential(
            *[ResBlock(channels=num_filters) for _ in range(num_res_blocks)]
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )




    def forward(self, x):
        first = self.input_conv(x)
        second = self.res_tower(first)
        third = self.value_head(second)
        return third

