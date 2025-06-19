import torch
from torch import nn
import torchvision
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, ot_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ot_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ot_ch, ot_ch, 3)
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

block = Block(1, 64)
x = torch.randn(1,1,572,572)
print(block(x).shape)

class Encoder(nn.Module):
    def __init__(self, chs = (3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs) - 1)])
        self.maxpool = nn.MaxPool2d(2)
    def forward(self, x):
        fltrs = []
        for block in self.blocks:
            x = block(x)
            fltrs.append(x)
            x = self.maxpool(x)
        return fltrs
            
encoder = Encoder()
x = torch.randn(1,3,572,572)
fltrs = encoder(x)
for flt in fltrs:
    print(flt.shape)

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs) - 1)])
    def forward(self, x, enc_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_fltrs = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_fltrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_fltrs, x):
        _, _, H, W = x.shape
        enc_fltrs = torchvision.transforms.CenterCrop([H, W])(enc_fltrs)
        return enc_fltrs


decoder = Decoder()
x = torch.randn(1,1024,28,28)
print(decoder(x, fltrs[::-1][1:]).shape)


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=True):
        super().__init__()
        self.out_sz = (572,572)
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out  = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

unet = UNet()
x = torch.randn(1,3,572,572)

output = unet(x)







