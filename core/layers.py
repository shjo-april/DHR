import torch
from torch import nn
from torch.nn import functional as F

from .backbones import resnet
from .backbones import wide_resnet

class Backbone(nn.Module):
    def __init__(self, backbone, norm_fn='bn', act_fn='relu', pretrained=True, last_stride=2, output_stride=16):
        super().__init__()

        if norm_fn == 'bn': self.norm_fn = nn.BatchNorm2d
        if act_fn == 'relu': self.act_fn = lambda:nn.ReLU(inplace=True)

        if 'wide' in backbone:
            self.model = wide_resnet.build_wide_resnet(backbone, last_stride, pretrained, freeze=True)
            self.in_channels = [128, 256, 512, 1024, 4096]
        elif 'resnet' in backbone:
            self.model = resnet.build_resnet(backbone, self.norm_fn, self.act_fn, last_stride, pretrained, output_stride)
            self.in_channels = [64, 256, 512, 1024, 2048]
        
    def get_parameters(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name:
                if 'weight' in name: groups[0].append(value)
                else: groups[1].append(value)
            # scracthed weights
            else:
                if 'weight' in name: groups[2].append(value)
                else:groups[3].append(value)
        return groups

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding=0, dilation=1, bias=False, dropout=0., norm=nn.BatchNorm2d, act=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.bn = norm(planes) if norm is not None else nn.Identity()
        self.act = act() if act is not None else nn.Identity()    
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class DeepLabv1_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, 512, 
            kernel_size=3, stride=1, 
            padding=12, dilation=12, bias=False
        )
        self.bn1 = nn.BatchNorm2d(512, momentum=0.0003)

        self.conv2 = nn.Conv2d(
            512, 512, 
            kernel_size=1, stride=1, 
            padding=0, dilation=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(512, momentum=0.0003)
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv2d(512, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.dropout(x)
        x = self.classifier(x)

        return x

class DeepLabv2_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        dilations = [6, 12, 18, 24]
        
        self.aspp1 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[0], dilation=dilations[0], bias=True
        )
        self.aspp2 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[1], dilation=dilations[1], bias=True
        )
        self.aspp3 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[2], dilation=dilations[2], bias=True
        )
        self.aspp4 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[3], dilation=dilations[3], bias=True
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        return x1 + x2 + x3 + x4

class DeepLabv3_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super().__init__()
        
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = ConvBlock(in_channels, out_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ConvBlock(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ConvBlock(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ConvBlock(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(in_channels, out_channels, 1)
        )
        self.block = ConvBlock(out_channels * 5, out_channels, 1, dropout=0.5)
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.gap(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.block(x)

        return x
    
class DeepLabv3plus_Head(nn.Module):
    def __init__(self, in_channels, out_channels, low_in_channels, low_out_channels, output_stride, num_classes):
        super().__init__()

        self.aspp = DeepLabv3_ASPP(in_channels, out_channels, output_stride)
        
        self.low_block = ConvBlock(low_in_channels, low_out_channels, 1)
        self.mid_block = nn.Sequential(
            ConvBlock(out_channels + low_out_channels, out_channels, kernel_size=3, padding=1, dropout=0.5),
            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1, dropout=0.1),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x, x_low):
        x = self.aspp(x)
        x_low = self.low_block(x_low)

        x_seg = F.interpolate(x, size=x_low.size()[2:], mode='bilinear', align_corners=False)
        x_seg = torch.cat((x_seg, x_low), dim=1)

        x_dec = self.mid_block(x_seg)
        return self.classifier(x_dec), x_dec
