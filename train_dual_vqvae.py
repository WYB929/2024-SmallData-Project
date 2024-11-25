import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from animal_dataset import AnimalDataset 
from models.dual_vq_autoencoder import DualVQVAE  
from models.audio_encoder import ConvolutionalAudioEncoder
from models.audio_decoder import ConvolutionalAudioDecoder
from models.encoder import ResNetEncoder
from models.decoder import ResNetDecoder
from tqdm import tqdm
def train_dual_vqvae(model, dataloader, optimizer, device, epochs=10, alpha=10, beta=1.0):
    """
    Train the DualVQVAE model.

    Args:
        model (nn.Module): DualVQVAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        optimizer (Optimizer): Optimizer for training.
        device (torch.device): Device for training (CPU or GPU).
        epochs (int): Number of training epochs.
        alpha (float): Weight for the VQ loss.
        beta (float): Weight for the NT-Xent loss.
    """
    model.to(device)
    model.train()

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        total_rec_audio_loss = 0.0
        total_rec_image_loss = 0.0
        total_vq_audio_loss = 0.0
        total_vq_image_loss = 0.0
        total_nt_xent_loss = 0.0

        for batch in tqdm(dataloader):
            # Load data
            images, audios, _ = batch  # Label is not used for training
            images, audios = images.to(device), audios.to(device)

            # Forward pass and compute loss
            loss, rec_audio_loss, rec_image_loss, vq_audio_loss, vq_image_loss, nt_xent_loss, _, _ = model.compute_loss(
                audios, images, alpha=alpha, beta=beta
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses for logging
            total_loss += loss.item()
            total_rec_audio_loss += rec_audio_loss.item()
            total_rec_image_loss += rec_image_loss.item()
            total_vq_audio_loss += vq_audio_loss.item()
            total_vq_image_loss += vq_image_loss.item()
            total_nt_xent_loss += nt_xent_loss.item()

        # Log epoch statistics
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"  Total Loss: {total_loss:.4f}")
        print(f"  Rec Audio Loss: {total_rec_audio_loss:.4f}")
        print(f"  Rec Image Loss: {total_rec_image_loss:.4f}")
        print(f"  VQ Audio Loss: {total_vq_audio_loss:.4f}")
        print(f"  VQ Image Loss: {total_vq_image_loss:.4f}")
        print(f"  NT-Xent Loss: {total_nt_xent_loss:.4f}")

# Main training logic
if __name__ == "__main__":
    # Define hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    epochs = 10
    alpha = 10
    beta = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Initialize dataset and dataloader
    dataset = AnimalDataset("/data2/zijin/random/2024-SmallData-Project/training_dataset")  # Ensure AnimalDataset is correctly implemented
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model
    audio_encoder = ConvolutionalAudioEncoder()
    audio_decoder = ConvolutionalAudioDecoder()
    image_encoder = ResNetEncoder()
    image_decoder = ResNetDecoder()
    model = DualVQVAE(audio_encoder, audio_decoder, image_encoder, image_decoder)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_dual_vqvae(model, dataloader, optimizer, device, epochs=epochs, alpha=alpha, beta=beta)
