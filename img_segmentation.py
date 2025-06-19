import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, in_ch, ot_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ot_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ot_ch, ot_ch, 3)  # conv2 is an instance of class Conv2d
    def forward(self, x):
       return self.relu(self.conv2(self.relu(self.conv1(x))))   # self.conv2.__call__(x)  

enc_block = Block(1, 64)
x  = torch.randn(1, 1, 572, 572)
print(enc_block(x).shape)

class Encoder(nn.Module):
    def __init__():
        super().__init__()
        

