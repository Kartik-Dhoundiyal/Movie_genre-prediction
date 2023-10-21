import torch
import torch.nn as nn
from torchvision import models

# Define the self-attention module
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Required shape: (seq_len, batch_size, embed_dim)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  # Return to original shape: (batch_size, seq_len, embed_dim)
        x = self.norm(x)
        return x

# Define the custom ResNet-50 model with attention
def model_with_attention(pretrained, requires_grad, num_attention_heads):
    # Load ResNet-50
    model = models.resnet50(pretrained=pretrained)
    
    # Modify the final classification layer
    model.fc = nn.Linear(2048, 25)
    
    # Add the attention layer after the final convolutional layer
    model.layer4.add_module('self_attention', SelfAttention(2048, num_attention_heads))
    
    # Freeze or unfreeze layers as needed
    for param in model.parameters():
        param.requires_grad = requires_grad
    
    return model

# Example usage
num_attention_heads = 8  # You can adjust this number based on your needs
custom_model = model_with_attention(pretrained=True, requires_grad=True, num_attention_heads=num_attention_heads)
