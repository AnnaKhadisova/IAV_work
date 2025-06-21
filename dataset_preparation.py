import torch
from torch import nn
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Setup path to data folder
image_path = Path("tusimple/")




# def walk_through_dir(dir_path):
#   for dirpath, dirnames, filenames in os.walk(dir_path):
#     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# walk_through_dir(image_path)

# Setup train and testing paths
train_dir = image_path / "train" / "frames"
test_dir = image_path / "test" / "frames"


print(train_dir)

# visualise an image: 1) they are extracting random image, 2) extracting their label, 3) width and height
import random
from PIL import Image
from numpy import asarray
# Get all image paths (* means "any combination")
image_path_list = list(train_dir.glob("*.jpg"))

# # 2. Get random image path
random_image_path = random.choice(image_path_list)
img = Image.open(random_image_path)
# display the image
#img.show()
print(random_image_path)

# 3. Get image label from path name
# 1. Get the image's name  2. Get the image with this name from train/lane_masks folder 3. Convert this into a mask
file_name = os.path.basename(random_image_path)
#print(file_name)

file_label = Image.open(image_path / 'train' / 'lane_masks' / file_name)
random_image_label = asarray(file_label)
#print(random_image_label)


# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {random_image_label}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")


# # 1. Turn our data into tensors
# # 2. Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.DataLoader
from torchvision import transforms
# transform = transforms.ToTensor()
# image_tensor = transform(img)
# print(image_tensor.dtype)
# print(image_tensor.shape)

# Load image data with a custom dataset, write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset
from typing import Tuple, Dict, List

class ImageSegmentationDataset(Dataset):
  def __init__(self, targ_dir: str, transform=None) -> None:    # targ_dir = train_dir
    self.paths = list(targ_dir.glob("*.jpg")) # all images paths
    self.transform = transform
    
  def load_image(self, index: int):
    image_path = self.paths[index]
    return Image.open(image_path).convert("RGB")

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index:int):     # return one sample of data, data and label (X, y)
    img = self.load_image(index)
    file_name = os.path.basename(self.paths[index])
    label = Image.open(image_path / 'train' / 'lane_masks' / file_name).convert("L")
    if self.transform:
      return (self.transform(img), self.transform(label))
    else:
      return (img, label)
  

# checking if we can receive the tuple of data and overlay mask onto the image
import numpy as np
instance = ImageSegmentationDataset(train_dir)
print("Class success", instance.paths[0])
instance.load_image(50)
tuple_ = instance.__getitem__(50)
tuple_[0].show()
tuple_[1].show()

image_np = np.array(tuple_[0])
mask_np = np.array(tuple_[1])


# Make mask red
colored_mask = np.zeros_like(image_np)
colored_mask[:, :, 0] = mask_np  # Red channel

# === Step 3: Overlay ===
overlayed = image_np.copy()
alpha = 0.5  # Transparency factor
overlayed = np.where(colored_mask > 0, 
                     (1 - alpha) * image_np + alpha * colored_mask, 
                     image_np).astype(np.uint8)

# plot
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 3, 1)
# plt.title("Raw Image")
# plt.imshow(image_np)
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Mask")
# plt.imshow(mask_np, cmap='gray')
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.title("Overlay")
# plt.imshow(overlayed)
# plt.axis("off")

# plt.tight_layout()
# plt.show()

transform = transforms.Compose([
  transforms.Resize((64,64)),
  transforms.ToTensor()
])
# image_tensor = transform(img)

# let's turn our images into Dataset using our castom class
train_data_custom = ImageSegmentationDataset(targ_dir = train_dir, transform=transform)
test_data_custom = ImageSegmentationDataset(targ_dir = test_dir, transform=transform)

# checking functions
print(train_data_custom, test_data_custom)
print(len(train_data_custom),len(test_data_custom))

########### now we have the dataset with type: torch.utils.data.dataset.Dataset ################

### turn into Dataloader's, using torch.utils.data.DataLoader()
from torch.utils.data import DataLoader
train_dataloader_custom = DataLoader(dataset=train_data_custom, batch_size=64, num_workers=0, shuffle=False)
test_dataloader_custom = DataLoader(dataset=test_data_custom, batch_size=64, num_workers=0, shuffle=False)

print(train_dataloader_custom, test_dataloader_custom)

# working with dataloaders
# get image and label from custom DataLoader
img_custom, label_custom = next(iter(train_dataloader_custom))
print(len(train_dataloader_custom))
print(img_custom.shape)
print(label_custom.shape)

# create u-net model class
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
print(unet)


# try a forward pass on a single image
img_batch, label_batch = next(iter(train_dataloader_custom)) # a batch of images and labels
