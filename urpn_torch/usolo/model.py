from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional

from urpn_torch.unet import model
from urpn_torch.usolo import cood_conv


class URPN(model.UNet):
    def __init__(self, in_channels=3, features=32, cells: Tuple[int, int] = (5, 5)):
        super(URPN, self).__init__(in_channels, features)

        self.encoder5 = URPN._block(features * 8, features * 16, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder6 = URPN._block(features * 16, features * 16, name="enc6")
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = URPN._block(features * 16, features * 16, name="bottleneck")
        
        self.upconv6 = nn.ConvTranspose2d(features * 16, features * 16, kernel_size=2, stride=2)
        self.decoder6 = URPN._block((features * 16) * 2, features * 16, name="dec6")

        self.upconv5 = nn.ConvTranspose2d(features * 16, features * 16, kernel_size=2, stride=2)
        self.decoder5 = URPN._block((features * 16) * 2, features * 8, name="dec5")
        
        # Mask branch
        self.upconv4 = nn.ConvTranspose2d(features * 8, features * 8, kernel_size=2, stride=2)
        self.decoder4 = URPN._block_coord_conv((features * 8) * 2, features * 8, name="enc1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=(cells[0] * cells[1]),
                              kernel_size=1)

        # Class branch
        self.cls_1 = super()._block(features * 16, features * 16, name="cls_1")

        self.cls_2 = super()._block(features * 16, features * 16, name="cls_2")

        self.cls_3 = super()._block(features * 16, features * 8, name="cls_3")

        self.cls_4 = super()._block(features * 8, features * 4, name="cls_4")
        
        self.cls_5 = super()._block(features * 4, features * 2, name="cls_5")
            
        self.cls_6 = super()._block(features * 2, features, name="cls_6")

        self.cls_out = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1)

    @staticmethod
    def _block_coord_conv(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",
                     cood_conv.CoordConv(
                         in_channels=in_channels,
                         out_channels=features,
                         kernel_size=3,
                         padding=1,
                         bias=False,
                     ),
                     ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2",
                     nn.Conv2d(
                         in_channels=features,
                         out_channels=features,
                         kernel_size=3,
                         padding=1,
                         bias=False,
                     ),
                     ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))

        bottleneck = self.bottleneck(self.pool6(enc6))

        dec6 = self.upconv6(bottleneck)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)
                
        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        
        
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        mask = torch.sigmoid(self.conv(dec1))
        
        aligned_cls = functional.interpolate(bottleneck, (5, 5))
        
        cls1 = self.cls_1(aligned_cls)
        cls2 = self.cls_2(cls1)
        cls3 = self.cls_3(cls2)
        cls4 = self.cls_4(cls3)
        cls5 = self.cls_5(cls4)
        cls6 = self.cls_6(cls5)        

        classification = self.cls_out(cls6)

        return mask, classification



