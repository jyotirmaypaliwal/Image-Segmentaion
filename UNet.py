 def double_conv(in_c, out_c):
     conv = nn.Sequential(
         nn.Conv2d(in_c, out_c, kernel_size=3),
         nn.ReLU(inplace=True),
         nn.Conv2d(out_c, out_c, kernel_size=3),
         nn.ReLU(inplace=True))
     return conv
    
    
    
def crop_img(tensor, target_tensor):
     target_size = target_tensor.size()[2]
     tensor_size = tensor.size()[2]
     delta = tensor_size - target_size
     delta = delta // 2
     return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UNet(nn.Module):
     def __init__(self):
         super(UNet, self).__init__()
        
         self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride = 2)
         self.down_conv_1 = double_conv(3, 64)
         self.b1 = nn.BatchNorm2d(64)
         self.down_conv_2 = double_conv(64, 128)
         self.b2 = nn.BatchNorm2d(128)

         self.down_conv_3 = double_conv(128, 256)
         self.b3 = nn.BatchNorm2d(256)

         self.down_conv_4 = double_conv(256, 512)
         self.b4 = nn.BatchNorm2d(512)

         self.down_conv_5 = double_conv(512, 1024)
         self.b5 = nn.BatchNorm2d(1024)

        
        
         self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,
                                             out_channels=512,
                                             kernel_size = 2,
                                             stride=2)
        
         self.up_conv_1 = double_conv(1024, 512)
         self.b6 = nn.BatchNorm2d(512)

        
        
         self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,
                                             out_channels=256,
                                             kernel_size = 2,
                                             stride=2)
        
         self.up_conv_2 = double_conv(512, 256)
         self.b7 = nn.BatchNorm2d(256)

        
        
         self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size = 2,
                                             stride=2)
        
         self.up_conv_3 = double_conv(256, 128)
         self.b8 = nn.BatchNorm2d(128)

        
        
         self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size = 2,
                                             stride=2)
        
         self.up_conv_4 = double_conv(128, 64)
         self.b9 = nn.BatchNorm2d(64)

        
         self.out = nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1)
        
        
        
        
        
        
     def forward(self, image):
         x1 = self.down_conv_1(image)
         x1 = self.b1(x1)
         x2 = self.max_pool_2x2(x1)
        
        
         x3 = self.down_conv_2(x2)
         x3 = self.b2(x3)
         x4 = self.max_pool_2x2(x3)
        
        
         x5 = self.down_conv_3(x4)
         x5 = self.b3(x5)
         x6 = self.max_pool_2x2(x5)
        
        
         x7 = self.down_conv_4(x6)
         x7 = self.b4(x7)
         x8 = self.max_pool_2x2(x7)
        
        
         x9 = self.down_conv_5(x8)
         x9 = self.b5(x9)
        
         x = self.up_trans_1(x9)
         y = crop_img(x7, x)
         x = self.up_conv_1(torch.cat([x, y], 1))
         x = self.b6(x)
        
         x = self.up_trans_2(x)
         y = crop_img(x5, x)
         x = self.up_conv_2(torch.cat([x, y], 1))
         x=self.b7(x)
        
         x = self.up_trans_3(x)
         y = crop_img(x3, x)
         x = self.up_conv_3(torch.cat([x, y], 1))
         x = self.b8(x)
        
        
         x = self.up_trans_4(x)
         y = crop_img(x1, x)
         x = self.up_conv_4(torch.cat([x, y], 1))
         x = self.b9(x)
        
         return x