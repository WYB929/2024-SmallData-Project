import torch
import torch.nn as nn
from einops import rearrange
from vector_quantize_pytorch import ResidualVQ
import torch.nn.functional as F


class DualVQVAE(nn.Module):
    def __init__(
        self,
        audio_encoder,
        audio_decoder,
        image_encoder,
        image_decoder,
        audio_h_dim=512,
        image_h_dim=512,
        num_quantizers=2,
        codebook_size=1024,
        temperature=0.07,
    ):
        """
        Args:
            audio_encoder (nn.Module): Encoder module for audio.
            audio_decoder (nn.Module): Decoder module for audio.
            image_encoder (nn.Module): Encoder module for image.
            image_decoder (nn.Module): Decoder module for image.
            audio_h_dim (int): Dimensionality of the latent space for audio.
            image_h_dim (int): Dimensionality of the latent space for image.
            num_quantizers (int): Number of quantizers for VQ.
            codebook_size (int): Size of the VQ codebook.
            temperature (float): Temperature for NT-Xent loss.
        """
        super(DualVQVAE, self).__init__()
        self.audio_encoder = audio_encoder
        self.audio_vq = ResidualVQ(
            dim=audio_h_dim, codebook_size=codebook_size, num_quantizers=num_quantizers
        )
        self.audio_decoder = audio_decoder

        self.image_encoder = image_encoder
        self.image_vq = ResidualVQ(
            dim=image_h_dim, codebook_size=codebook_size, num_quantizers=num_quantizers
        )
        self.image_decoder = image_decoder

        self.temperature = temperature

    def forward(self, audio_input, image_input):
        # Process audio
        z_audio = self.audio_encoder(audio_input)
        z_audio_flat = rearrange(
            z_audio, "b c l -> b l c"
        )  # Flatten for vector quantization
        vq_audio, audio_indices, vq_audio_loss = self.audio_vq(z_audio_flat)
        vq_audio = rearrange(vq_audio, "b l c -> b c l")  # Reshape back for decoder
        recon_audio = self.audio_decoder(vq_audio)

        # Process image
        z_image = self.image_encoder(image_input)
        z_image_flat = rearrange(
            z_image, "b d h w -> b (h w) d"
        )  # Flatten for vector quantization
        vq_image, image_indices, vq_image_loss = self.image_vq(z_image_flat)
        vq_image = rearrange(
            vq_image, "b (h w) d -> b d h w", h=z_image.size(-2), w=z_image.size(-1)
        )
        recon_image = self.image_decoder(vq_image)

        return (
            recon_audio,
            recon_image,
            vq_audio_loss,
            vq_image_loss,
            audio_indices,
            image_indices,
        )

    def compute_nt_xent_loss(self):
        """
        Computes the NT-Xent contrastive loss between the audio and image codebooks.

        Returns:
            torch.Tensor: NT-Xent loss.
        """
        # Extract codebook embeddings from both vector quantizers
        audio_codebook = rearrange(self.audio_vq.codebooks, "n q d -> (n q) d")
        image_codebook = rearrange(self.image_vq.codebooks, "n q d -> (n q) d")

        # Normalize embeddings
        audio_codebook = F.normalize(audio_codebook, dim=-1)
        image_codebook = F.normalize(image_codebook, dim=-1)

        # Cosine similarity matrix
        similarity_matrix = (
            torch.matmul(audio_codebook, image_codebook.T) / self.temperature
        )

        # Positive samples are along the diagonal
        # TODO try only be the choosen code for sample?
        labels = torch.arange(similarity_matrix.size(0), device=audio_codebook.device)

        # Contrastive loss
        loss_audio_to_image = F.cross_entropy(similarity_matrix, labels)
        loss_image_to_audio = F.cross_entropy(similarity_matrix.T, labels)
        return (loss_audio_to_image + loss_image_to_audio) / 2

    def compute_loss(self, audio_input, image_input, alpha=10, beta=0.0001):
        """
        Computes the reconstruction loss, vector quantization loss, and NT-Xent loss.

        Args:
            audio_input (torch.Tensor): Input batch of audio signals.
            image_input (torch.Tensor): Input batch of images.
            alpha (float): Weight for the VQ loss.
            beta (float): Weight for the NT-Xent loss.

        Returns:
            tuple: Total loss, reconstruction losses, VQ losses, NT-Xent loss, and indices.
        """
        (
            recon_audio,
            recon_image,
            vq_audio_loss,
            vq_image_loss,
            audio_indices,
            image_indices,
        ) = self.forward(audio_input, image_input)

        # Clamp outputs to valid ranges
        recon_audio = recon_audio.clamp(-1.0, 1.0)
        recon_image = recon_image.clamp(-1.0, 1.0)

        # Reconstruction losses
        rec_audio_loss = (recon_audio - audio_input).abs().mean()
        rec_image_loss = (recon_image - image_input).abs().mean()

        # NT-Xent loss
        nt_xent_loss = self.compute_nt_xent_loss()

        # Total loss
        total_rec_loss = rec_audio_loss + rec_image_loss
        total_vq_loss = vq_audio_loss.sum() + vq_image_loss.sum()
        total_loss = total_rec_loss + alpha * total_vq_loss + beta * nt_xent_loss

        return (
            total_loss,
            rec_audio_loss,
            rec_image_loss,
            vq_audio_loss.sum(),
            vq_image_loss.sum(),
            nt_xent_loss,
            audio_indices,
            image_indices,
        )
