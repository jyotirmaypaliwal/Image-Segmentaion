# System 
import os
import sys

# Data and image processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import cv2
from PIL import Image
import albumentations as A

# Pytorch libs
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset ,DataLoader
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy



# Creating variables
train_directory= "/kaggle/input/cityscapes-image-pairs/cityscapes_data/train/"
val_directory = "/kaggle/input/cityscapes-image-pairs/cityscapes_data/val/"
train_img_names = os.listdir(train_directory)
val_img_names = os.listdir(val_directory)


# Kmeans 
# Preparing masks (segmetation for Kmeans)
kmeans_data = []
for img in train_img_names[0:50]:
    org_img = cv2.imread(os.path.join(train_directory, img))
    imgg = org_img[:, 0:256,:]
    msk = org_img[:, 256:,:]
    
    kmeans_data.append(msk)
kmeans_data = np.array(kmeans_data)
kmeans_data = kmeans_data.reshape(-1,3)
print(kmeans_data.shape)

# Importing and applying Kmeans
from sklearn.cluster import KMeans
encoder = KMeans(n_clusters=8)
encoder.fit(kmeans_data)

# Encoding
for img in train_img_names[0:2]:
    org_img = cv2.imread(os.path.join(train_directory, img))
    msk = org_img[:, 256:,:]
    test = msk.reshape(-1,3)
    pred = encoder.predict(test)
    
    enc_pred = pred.reshape(256, 256)
    
    pred = np.array([colors[p] for p in pred]).reshape(256,256,3)
    
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(msk)
    plt.title('Original mask (RGB)')
    plt.subplot(1,2,2)
    plt.imshow(pred)
    plt.title('Encoded mask')
    plt.show()



# Creating Dataset class
class Dataset(Dataset):
    def __init__(self, images_list, size, mean = None, std = None):
        self.images_list = images_list
        self.size = size
        
        if mean is None or std is None:
            self.mean = [0., 0., 0.]
            self.std = [1., 1., 1.]
        else:
            self.mean = mean
            self.std = std
        
    def __len__(self):
        return len(os.listdir(self.images_list))

    def __getitem__(self, index):
        img = os.listdir(self.images_list)[index]
        pth = os.path.join(self.images_list, img)

        
          
        pil_image = Image.open(pth).convert('RGB')
        org_img = np.array(pil_image)
        
        np_image = org_img[:, 0:256,:]
        np_target = org_img[:, 256:,:] 
        
        test = (np_target.reshape(-1,3))      
        pred = encoder.predict(test)
        seg_msk = pred.reshape(256,256)
        
        
        
        trans_obj = A.Compose([A.Resize(self.size, self.size),
                                   A.Normalize(self.mean, self.std)])
            
        transformed = trans_obj(image = np_image, mask = seg_msk)
        img_tensor = from_numpy(transformed['image']).permute(2, 0, 1)
        mask_tensor = from_numpy(transformed['mask'])
        return img_tensor, mask_tensor



# Dataloader
train_data_obj = Dataset(train_directory, 256, mean=None, std=None)
train_dataloader = DataLoader(train_data_obj, batch_size=17, shuffle=True )

test_data_obj = Dataset(val_directory, 256, mean=None, std=None)
test_dataloader = DataLoader(test_data_obj, batch_size=10, shuffle=True)

test1_data_obj = Dataset(val_directory, 256, mean = None, std=None)
test1_dataloader = DataLoader(test1_data_obj, batch_size=1, shuffle=True)



class UNetTunable(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        conv_mode='standard'
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        
        super(UNetTunable, self).__init__()
        assert conv_mode in ('standard', 'dilated')
        assert up_mode in ('upconv', 'upsample', 'dilated')
        
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm,conv_mode)
            ).to(device)
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm,conv_mode)
            ).to(device)
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm,conv_mode):
        super(UNetConvBlock, self).__init__()
        
        if conv_mode == 'standard':
            block = []
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            self.block = nn.Sequential(*block)
            
        elif conv_mode == 'dilated':
            block = []
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding), dilation = 1 ))
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding), dilation = 1 ))
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, conv_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        elif up_mode == 'dilated':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, conv_mode)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out



# Instantiating model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetTunable(in_channels=3,
                        n_classes=8,
                        depth=5,
                        wf=6,
                        padding=True,
                        batch_norm=True,
                        up_mode='upconv',
                        conv_mode='dilated'
                       ).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# Optimizer
import torch.optim 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


# Loss
def compute_loss(y_hat, y):
    return F.cross_entropy(y_hat, y)


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(50):
    total_loss = 0
    
    for batch in train_dataloader:
        image, msk = batch
        mski = transforms.Resize(256)
        msk = mski(msk).long()


        

        image = image.to(device=device)
        label = msk.to(device=device)

        model.to(device)
        predicted = model(image)
        loss = compute_loss(predicted, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print("epoch: ", epoch, " loss: ", total_loss)



# Validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tl=0
total_correct=0
for batch in test_dataloader:
    imagi, mask = batch
    maskkk = transforms.Resize(256)
    mskk = maskkk(mask).long()
    
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.to(device);
    image = imagi.to(device=device)
    labeli = mskk.to(device=device)
    
    predictionss = model(image)
    loss = compute_loss(predictionss, labeli) 
    
    tl += loss.item()
    
print("Total loss: ", tl)


# Visualizing results
# i=0
# for batch in test1_dataloader:
#     imagee, msok = batch
#     mskie = transforms.Resize(256)
#     mskl = mskie(msok).long()
    
#     model.to(device)
#     imagee = imagee.to(device=device)
#     masku = mskl.to(device=device)
#     p1 = model(imagee)
#     p1 = p1[0].squeeze(dim=0).permute(1,2,0)
#     p1 = torch.argmax(p1, dim=2)
#     print(p1.shape)
#     i+=1
#     if i > 3:
#         break
#     imagee = torch.Tensor.cpu(imagee)
#     p1 = torch.Tensor.cpu(p1)
#     masku = torch.Tensor.cpu(masku)
       
#     plt.figure(figsize=(10,10))
#     plt.subplot(1,3,1)
#     plt.imshow(imagee.squeeze(dim=0).permute(1,2,0))
#     plt.title('Original image')
#     plt.subplot(1,3,2)
#     plt.imshow(masku.permute(1,2,0))
#     plt.title('Encoded mask')
#     plt.subplot(1,3,3)
#     plt.imshow(p1)
#     plt.title('Predicted mask')
#     plt.show()




