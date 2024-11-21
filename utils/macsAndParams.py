from thop import profile
from model.unet import UNet
from model.CUnet import CUNet
from model.UNetPlusPlus import NestedUNet
from model.res_unet_plus import ResUnetPlusPlus
import torch
from torchvision.models import resnet50

model = resnet50()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))

model = UNet(3, 2)
input = torch.randn(1, 3, 224, 224)
macs1, params1 = profile(model, inputs=(input, ))


model = CUNet(3, 2)
input = torch.randn(1, 3, 224, 224)
macs2, params2 = profile(model, inputs=(input, ))

model = NestedUNet(3, 2)
input = torch.randn(1, 3, 224, 224)
macs3, params3 = profile(model, inputs=(input, ))

model = ResUnetPlusPlus(3, 2)
input = torch.randn(1, 3, 224, 224)
macs4, params4 = profile(model, inputs=(input, ))

print("macs: {}, params: {}".format(macs, params))
print("macs: {}, params: {}".format(macs1, params1))
print("macs: {}, params: {}".format(macs2, params2))
print("macs: {}, params: {}".format(macs3, params3))
print("macs: {}, params: {}".format(macs4, params4))