import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from animal_dataset import AnimalDataset  # Assuming this is the dataset class
from models.audio_vq_autoencoder import SimpleVQAutoEncoder
from models.audio_encoder import ConvolutionalAudioEncoder
from models.audio_decoder import ConvolutionalAudioDecoder


def train_vqvae(model, dataloader, optimizer, device, num_epochs=20, log_interval=10):
    """
    Train the VQ-VAE model on the given dataset.

    Args:
        model (nn.Module): The VQ-VAE model.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        log_interval (int): Logging interval for training progress.
    """
    model.to(device)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (_, audio, _) in enumerate(dataloader):  # Extract audio only
            audio = audio.to(device)

            # Ensure input is of shape [batch_size, channels, sequence_length]
            if audio.ndim == 2:
                audio = audio.unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            total_loss, rec_loss, vq_loss, _ = model.compute_loss(audio)

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            if batch_idx % log_interval == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}]: "
                    f"Reconstruction Loss: {rec_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}"
                )

        print(f"Epoch [{epoch}/{num_epochs}] completed. Average Loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    # Configuration
    dataset_path = "/data2/zijin/random/2024-SmallData-Project/training_dataset"
    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 20
    log_interval = 10

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = AnimalDataset(root_dir=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model setup
    encoder = ConvolutionalAudioEncoder(in_dim=1)  # Mono audio input
    decoder = ConvolutionalAudioDecoder(out_dim=1)
    model = SimpleVQAutoEncoder(encoder=encoder, decoder=decoder, h_dim=128, num_quantizers=3, codebook_size=1024)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_vqvae(model, dataloader, optimizer, device, num_epochs=num_epochs, log_interval=log_interval)

    # Save the model
    save_path = "audio_vqvae.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
