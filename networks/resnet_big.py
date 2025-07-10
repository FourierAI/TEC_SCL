"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import networks.model as model

model_dict = {
    'resnet18': [models.resnet18, 512],
    'resnet34': [models.resnet34, 512], 
    'resnet50': [models.resnet50, 2048],
    'resnet101': [models.resnet101, 2048],
    'resnet152': [models.resnet152, 2048],
    'senet154': [timm.models.senet154, 2048],
    'efficientnet': [timm.models.efficientnet.tf_efficientnetv2_m, 1280],
    'tec': [model.TEC, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet152', head='mlp', feat_dim=128, weight=False):
        super(SupConResNet, self).__init__()
        
        # Get model function and dimension
        model_fun, dim_in = model_dict[name]
        
        # Initialize encoder from torchvision with optional pretrained weights
        backbone = model_fun(pretrained=weight)

        self.encoder = backbone

        # Projection head
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x, sentences, masks):
        feat = self.encoder(x, sentences.squeeze(), masks)
        feat = torch.flatten(feat, 1)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10, weight=False):
        super(SupCEResNet, self).__init__()
        
        # Get base model
        model_fun, dim_in = model_dict[name]
        backbone = model_fun(pretrained=weight)
        
        # Remove original FC
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        
        # Add new classifier
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        if features.dim() > 3:
            features = features.squeeze()
        return self.fc(features)
