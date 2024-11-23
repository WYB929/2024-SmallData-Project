import torch
import torch.nn as nn
from einops import rearrange
from vector_quantize_pytorch import ResidualVQ
from models.encoder import Encoder
from models.decoder import Decoder


class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, in_dim=3, h_dim=256, res_h_dim=64, n_res_layers=3, num_quantizers=3, codebook_size=1024):
        super(SimpleVQAutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim)
        
        # Residual VQ
        self.vq = ResidualVQ(dim=h_dim, codebook_size=codebook_size, num_quantizers=num_quantizers)
        
        # Decoder
        self.decoder = Decoder(h_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        z_flat = rearrange(z, 'b d h w -> b (h w) d')  # Shape: [batch_size, h*w, d]
        
        # Vector Quantization
        vq_out, indices, vq_loss = self.vq(z_flat)
        print(vq_out.shape)
        # Reshape back to 2D feature map
        vq_out = rearrange(vq_out, 'b (h w) d -> b d h w', h=z.size(-2), w=z.size(-1))
        out = self.decoder(vq_out)
        return out, indices, vq_loss

    def compute_loss(self, x, alpha=10):
        """
        Computes the reconstruction loss and vector quantization loss.

        Args:
            x (torch.Tensor): Input batch of images.
            alpha (float): Weight for the VQ loss.

        Returns:
            tuple: Total loss, reconstruction loss, VQ loss, and indices.
        """
        out, indices, vq_loss = self.forward(x)
        out = out.clamp(-1., 1.)
        
        # Reconstruction Loss
        rec_loss = (out - x).abs().mean()
        
        # Weighted Total Loss
        total_loss = rec_loss + alpha * vq_loss.sum()
        
        return total_loss, rec_loss, vq_loss.sum(), indices
    
    def get_latent_vector(self, x):
        """
        Extracts the latent vector from the input.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: The latent vector (quantized representation).
        """
        # Encode the input
        z = self.encoder(x)
        
        # Flatten for vector quantization
        z_flat = rearrange(z, 'b d h w -> b (h w) d')  # Shape: [batch_size, h*w, d]
        
        # Vector Quantization
        vq_out, _, _ = self.vq(z_flat)
        return vq_out
