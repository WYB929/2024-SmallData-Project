import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        return self.conv_stack(x)


class ResNetEncoder(nn.Module):
    """
    Encoder module following the structure of ResNet-18, removing the fully connected layer.

    Inputs:
    - in_dim: Number of input channels (e.g., 3 for RGB images).
    """

    def __init__(self, in_dim=3):
        super(ResNetEncoder, self).__init__()
        self.resnet = torchvision.models.resnet18()

        # Modify the first convolutional layer to match input dimensions
        self.resnet.conv1 = nn.Conv2d(
            in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the average pool and FC layer
        self.resnet = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )

    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    # random data
    x = torch.randn(16, 3, 224, 224)  # Batch of 16 RGB images of size 224x224

# Initialize encoder
    encoder = ResNetEncoder(in_dim=3)

    # Forward pass
    encoder_out = encoder(x)
    print("Encoder output shape:", encoder_out.shape)  # Expected shape: [16, 512, 7, 7]