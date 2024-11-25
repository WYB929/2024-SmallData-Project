import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim=32):
        super(Decoder, self).__init__()
        h_dim=256
        n_res_layers=2
        res_h_dim=32
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class ResNetDecoder(nn.Module):
    """
    Transposed ResNet-style decoder.
    
    Inputs:
    - in_dim: Number of input channels (e.g., 512 for latent feature maps).
    - out_dim: Number of output channels (e.g., 3 for RGB images).
    """

    def __init__(self, in_dim=512):
        super(ResNetDecoder, self).__init__()

        # Layer configurations for transposed ResNet (reverse of ResNet-18)
        self.upsample_layers = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(in_dim, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Second upsampling block
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Third upsampling block
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Fourth upsampling block
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final upsampling block
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, x):
        return self.upsample_layers(x)


if __name__ == "__main__":
    # Example input tensor (latent feature map)
    x = torch.randn(16, 512, 7, 7)  # Batch of 16 latent feature maps of size 7x7

    # Initialize decoder
    decoder = ResNetDecoder(in_dim=512, out_dim=3)

    # Forward pass
    decoder_out = decoder(x)
    print("Decoder output shape:", decoder_out.shape)  # Expected shape: [16, 3, 224, 224]
