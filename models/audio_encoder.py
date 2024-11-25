import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A simple 1D residual block for feature extraction.
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


class ConvolutionalAudioEncoder(nn.Module):
    """
    Encoder module for audio signals.
    """

    def __init__(self, in_dim=1, num_hiddens=512, num_residual_layers=2, num_residual_hiddens=32):
        """
        Args:
            in_dim (int): Number of input channels (e.g., 1 for mono audio).
            num_hiddens (int): Number of hidden units for the convolutional layers.
            num_residual_layers (int): Number of residual blocks.
            num_residual_hiddens (int): Number of hidden units in residual blocks.
        """
        super(ConvolutionalAudioEncoder, self).__init__()

        # Preprocessing layers
        self.conv1 = nn.Conv1d(in_dim, num_hiddens // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_hiddens // 2, num_hiddens, kernel_size=3, stride=1, padding=1)

        # Downsampling layer
        self.conv3 = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input audio tensor of shape [batch_size, in_dim, seq_length].

        Returns:
            torch.Tensor: Encoded latent representation of the input.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.residual_blocks(x)
        return x


if __name__ == "__main__":
    # Example input tensor for testing (batch of mono audio signals with 16000 samples each)
    x = torch.randn(16, 1, 16000)  # Batch of 16 mono audio samples, each 16000 timesteps long

    # Initialize encoder
    encoder = ConvolutionalAudioEncoder(in_dim=1)

    # Forward pass
    encoder_out = encoder(x)
    print("Encoder output shape:", encoder_out.shape)
