from torch import nn
from torchvision.models import resnet18,resnet50
import torchvision
import torch

class WpodNet(nn.Module):

    def __init__(self):
        super(WpodNet, self).__init__()
        resnet = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        #resnet = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        backbone = list(resnet.children())
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(3),
            *backbone[:3],
            *backbone[4:8],
        )
        self.detection = nn.Conv2d(512, 8, 3, 1, 1)
        #self.detection = nn.Conv2d(2048, 8, 3, 1, 1)

    def forward(self, x):
        features = self.backbone(x)
        out = self.detection(features)
        #out = rearrange(out, 'n c h w -> n h w c')
        out = out.permute(0,2,3,1)
        return out


if __name__ == '__main__':
    m = WpodNet()
    x = torch.randn(2, 3, 256, 256)
    print(m(x).shape)#[2,16,16,8]  n,h,w,c
