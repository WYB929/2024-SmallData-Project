import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A simple 1D residual block for feature reconstruction.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)


class ConvolutionalAudioDecoder(nn.Module):
    """
    Decoder module for audio signals.
    """

    def __init__(self, out_dim=1, num_hiddens=512, num_residual_layers=2, num_residual_hiddens=32):
        """
        Args:
            out_dim (int): Number of output channels (e.g., 1 for mono audio).
            num_hiddens (int): Number of hidden units for the transposed convolutional layers.
            num_residual_layers (int): Number of residual blocks.
            num_residual_hiddens (int): Number of hidden units in residual blocks.
        """
        super(ConvolutionalAudioDecoder, self).__init__()

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        )

        # Upsampling layers
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,  # Upsample by factor of 2
            padding=1
        )
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=num_hiddens // 2,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Latent representation tensor of shape [batch_size, num_hiddens, seq_length].

        Returns:
            torch.Tensor: Reconstructed audio signal of shape [batch_size, out_dim, seq_length].
        """
        x = self.residual_blocks(x)
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv1(x))  # Normalize to [-1, 1]
        return x


if __name__ == "__main__":
    # Example latent input tensor for testing (batch of 16 latent feature maps)
    x = torch.randn(16, 128, 8000)  # Batch of 16, latent dim=128, downsampled length=8000

    # Initialize decoder
    decoder = ConvolutionalAudioDecoder(out_dim=1)

    # Forward pass
    decoder_out = decoder(x)
    print("Decoder output shape:", decoder_out.shape)  # Expected shape: [16, 1, 16000]
