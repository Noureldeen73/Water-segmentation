import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def createDeepLabv3(input_channels=8, output_channels=1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight 
        if input_channels > 3:
            mean_weight = old_conv.weight[:, :3].mean(dim=1, keepdim=True)
            for i in range(3, input_channels):
                new_conv.weight[:, i:i+1] = mean_weight

    model.backbone.conv1 = new_conv

    model.classifier = DeepLabHead(2048, output_channels)
    
    return model