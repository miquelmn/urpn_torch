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

        # Mask branch
        self.decoder4 = URPN._block_coord_conv((features * 8) * 2, features * 8, name="enc1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=(cells[0] * cells[1]),
                              kernel_size=1)

        # Class branch
        self.cls_1 = URPN._block(features * 16, features * 8, name="cls_1")
        # self.decoder4 = URPN._block((features * 8) * 2, features * 8, name="dec4")

        self.cls_2 = URPN._block(features * 8, features * 4, name="cls_2")
        # self.decoder3 = URPN._block((features * 4) * 2, features * 4, name="dec3")

        self.cls_3 = URPN._block(features * 4, features * 2, name="cls_3")
        # self.decoder2 = URPN._block((features * 2) * 2, features * 2, name="dec2")

        self.cls_4 = URPN._block(features * 2, features, name="cls_4")
        # self.decoder1 = URPN._block(features * 2, features, name="dec1")

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

        bottleneck = self.bottleneck(self.pool4(enc3))

        dec4 = self.upconv4(bottleneck)
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

        aligned_cls = functional.interpolate(bottleneck)
        cls1 = self.cls_1(aligned_cls)
        cls2 = self.cls_2(cls1)
        cls3 = self.cls_3(cls2)
        cls4 = self.cls_1(cls3)

        classification = self.cls_out(cls4)

        return mask, classification



