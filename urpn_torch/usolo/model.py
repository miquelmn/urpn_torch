from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional

from urpn_torch.unet import model
from urpn_torch.usolo import cood_conv


class URPN(model.UNet):
    def __init__(
        self,
        in_channels=3,
        features=32,
        cells: Tuple[int, int] = (5, 5),
        mask_size: Tuple[int, int] = (512, 512),
    ):
        super(URPN, self).__init__(in_channels, features)
        self.__mask_size = mask_size

        # Class branch
        self.cls_branch = URPN._branch(size=256, name="cls", depth=7)

        self.cls_out_1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.cls_out_2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.cls_out_3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.cls_out_4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.cls_out_5 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)

        self.mask_branch = URPN._branch(size=256, name="branch", depth=7)
        self.mask_out_1 = nn.Conv2d(
            in_channels=256, out_channels=(cells[0] * cells[1]), kernel_size=1
        )
        self.mask_out_2 = nn.Conv2d(
            in_channels=256, out_channels=(cells[0] * cells[1]), kernel_size=1
        )
        self.mask_out_3 = nn.Conv2d(
            in_channels=256, out_channels=(cells[0] * cells[1]), kernel_size=1
        )
        self.mask_out_4 = nn.Conv2d(
            in_channels=256, out_channels=(cells[0] * cells[1]), kernel_size=1
        )
        self.mask_out_5 = nn.Conv2d(
            in_channels=256, out_channels=(cells[0] * cells[1]), kernel_size=1
        )

    @staticmethod
    def _branch(size, name: str, depth: int):
        """Creates a SOLO branch.

        Args:
            size: Size of the features maps
            name: Name of the branch
            depth: Depth of the branch

        Returns:
            Pytorch Sequential of UNet blocks with depth and size defined by the parameters
        """
        layers = [
            (f"{name}_{level}", super()._block(size, size, name=f"{name}_{level}"))
            for level in range(1, depth)
        ]

        return nn.Sequential(OrderedDict([layers]))

    @staticmethod
    def _block_coord_conv(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
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
                    (
                        name + "conv2",
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
        bottleneck = self.bottleneck(self.pool4(enc4))

        # MASKS
        al_mask5 = functional.interpolate(bottleneck, self.__mask_size)
        mask5 = self.mask_branch(al_mask5)
        mask5 = self.mask_out_5(mask5)

        al_mask4 = functional.interpolate(enc4, self.__mask_size)
        mask4 = self.mask_branch(al_mask4)
        mask4 = self.mask_out_4(mask4)

        al_mask3 = functional.interpolate(enc3, self.__mask_size)
        mask3 = self.mask_branch(al_mask3)
        mask3 = self.mask_out_1(mask3)

        al_mask2 = functional.interpolate(enc2, self.__mask_size)
        mask2 = self.mask_branch(al_mask2)
        mask2 = self.mask_out_1(mask2)

        al_mask1 = functional.interpolate(enc1, self.__mask_size)
        mask1 = self.mask_branch(al_mask1)
        mask1 = self.mask_out_1(mask1)

        # Class

        al_cls5 = functional.interpolate(bottleneck, (5, 5))
        cls5 = self.cls_branch(al_cls5)
        cls5 = self.cls_out_5(cls5)

        al_cls4 = functional.interpolate(enc4, (5, 5))
        cls4 = self.cls_branch(al_cls4)
        cls4 = self.cls_out_4(cls4)

        al_cls3 = functional.interpolate(bottleneck, (5, 5))
        cls3 = self.cls_branch(al_cls3)
        cls3 = self.cls_out_3(cls3)

        al_cls2 = functional.interpolate(bottleneck, (5, 5))
        cls2 = self.cls_branch(al_cls2)
        cls2 = self.cls_out_2(cls2)

        al_cls1 = functional.interpolate(bottleneck, (5, 5))
        cls1 = self.cls_branch(al_cls1)
        cls1 = self.cls_out_1(cls1)

        mask = torch.cat([mask5, mask4, mask3, mask2, mask1], 1)
        classification = torch.cat([cls5, cls4, cls3, cls2, cls1], 1)

        return mask, classification
