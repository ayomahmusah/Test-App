import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F



ResNet50 = torchvision.models.resnet50()
ResNet50.fc = nn.Sequential(
    torch.nn.Linear(
        in_features=2048,
        out_features=8
    )
)