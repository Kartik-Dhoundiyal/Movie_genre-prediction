import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as models

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=2)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class AttentionModel(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(AttentionModel, self).__init__()
        self.resnet = models.resnet152(progress=True, pretrained=pretrained)

        # To freeze or train the hidden layers
        for param in self.resnet.parameters():
            param.requires_grad = requires_grad

        # Add attention mechanism
        self.attention = SelfAttention(2048)

        # Make the classification layer learnable
        # We have 25 classes in total
        self.resnet.fc = nn.Linear(2048, 25)

    def forward(self, x):
        x = self.resnet(x)
        x = self.attention(x)
        return x


