from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from model import CARAFE


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # self.conv = DoubleConv(in_channels, out_channels)
            self.up = CARAFE.CARAFE(in_channels)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Contrast(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Contrast, self).__init__()

        self.cvo_x = torch.Tensor([[[[-1, 1],
                                     [-2, 2]
                                     ]]])

        self.cvo_x = self.cvo_x.repeat(out_channels, in_channels, 1, 1)
        self.c_x = nn.Conv2d(in_channels, out_channels, (2, 2), stride=1, bias=False)
        self.c_x.weight.data = self.cvo_x

        self.cvo_y = torch.Tensor([[[[2, 1],
                                     [-2, -1]
                                     ]]])
        self.cvo_y = self.cvo_y.repeat(out_channels, in_channels, 1, 1)
        self.c_y = nn.Conv2d(in_channels, out_channels, (2, 2), stride=1, bias=False)
        self.c_y.weight.data = self.cvo_y

    def forward(self, x):
        with torch.no_grad():
            pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))
            x = pad(x)
            x_x = self.c_x(x)
            x_y = self.c_y(x)
            out = (abs(x_x) + abs(x_y)) ** 0.5

            return out


class CUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 16):
        super(CUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.se1 = SqueezeExcite(input_c=base_c, expand_c=base_c)
        # self.se1 = SqueezeExcite(base_c)

        self.down1 = Down(base_c, base_c*2)
        # self.cvo1 = Contrast(base_c, base_c)
        self.se2 = SqueezeExcite(input_c=base_c * 2, expand_c=base_c * 2)
        # self.se2 = SqueezeExcite(base_c*2)

        self.down2 = Down(base_c * 2, base_c * 2)
        self.cvo2 = Contrast(base_c * 2, base_c * 2)
        self.se3 = SqueezeExcite(input_c=base_c * 4, expand_c=base_c * 4)
        # self.se3 = SqueezeExcite(base_c*4)

        self.down3 = Down(base_c * 4, base_c * 4)
        self.cvo3 = Contrast(base_c * 4, base_c * 4)
        self.se4 = SqueezeExcite(input_c=base_c * 8, expand_c=base_c * 8)
        # self.se4 = SqueezeExcite(base_c*8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)


        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)

        x3 = self.down2(x2)
        out_1 = self.cvo2(x3)
        x3 = torch.cat((x3, self.cvo2(x3)), dim=1)



        x4 = self.down3(x3)

        #out_1 = self.cvo3(x4)
        out_2 = out_1[0]

        x4 = torch.cat((x4, self.cvo3(x4)), dim=1)
        x5 = self.down4(x4)
        x = self.up1(x5, self.se4(x4))
        x = self.up2(x, self.se3(x3))
        x = self.up3(x, self.se2(x2))
        x = self.up4(x, self.se1(x1))
        logits = self.out_conv(x)

        return logits


if __name__ == "__main__":
    image_path = "test.png"
    # 打开图像并转化为灰度图
    image = Image.open(image_path).convert('L')
    
    # 显示灰度图
    image.show()
    
    # 转化为Tensor对象
    tensor_image = torch.Tensor(list(image.getdata())).view(image.size[1], image.size[0])
    
    # 使用已有函数处理Tensor对象
    cvo_x = torch.Tensor([[[[-1, 1],
                                     [-2, 2]
                                     ]]])

    cvo_x = cvo_x.repeat(out_channels, in_channels, 1, 1)
    c_x = nn.Conv2d(in_channels, out_channels, (2, 2), stride=1, bias=False)
    c_x.weight.data = cvo_x

    cvo_y = torch.Tensor([[[[2, 1],
                                     [-2, -1]
                                     ]]])
    cvo_y = cvo_y.repeat(out_channels, in_channels, 1, 1)
    c_y = nn.Conv2d(in_channels, out_channels, (2, 2), stride=1, bias=False)
    c_y.weight.data = cvo_y
    
    # 将处理后的Tensor对象转化为灰度图
    processed_image = Image.fromarray(processed_tensor.numpy(), 'L')
    
    # 保存灰度图
    processed_image.save('processed_image.png')
    # import torch as t

    # rgb = t.randn(2, 3, 560, 560)

    # net = CUNet(3, 3)

    # out = net(rgb)

    # import torch
    #
    # rgb = torch.randn(2, 3, 560, 560)
    #
    # pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))
    # rgb = pad(rgb)
    #
    # c = torch.nn.Conv2d(3, 3, (2, 2), stride=1, bias=False)
    # w = torch.Tensor([[-1, 1],[-2, 2]]).unsqueeze(0).unsqueeze(0).repeat(cfg.BATCH_SIZE, 3, 1, 1)
    #
    # c = torch.Tensor([[[[-1, 1],
    #                                 [-2, 2],
    #                                 ]]]).repeat(cfg.BATCH_SIZE, 3, 1, 1)
    #
    # rgb1 = c(rgb)
    #
    # c2 = torch.nn.Conv2d(3, 3, (2, 2), stride=1, bias=False)
    # c2.weight.data = torch.Tensor([[[[2, 1],
    #                                 [-2, -1],
    #                                 ]]])
    # rgb2 = c2(rgb)
    # res = (abs(rgb1)+abs(rgb2))**0.5
    # pass
