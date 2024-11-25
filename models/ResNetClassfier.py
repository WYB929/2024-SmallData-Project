import torch
import torch.nn as nn
from models.encoder import ResNetEncoder


class ResNetClassifier(nn.Module):
    def __init__(self, encoder, num_classes, h_dim=512, loss_fn=None):
        """
        ResNet-based classifier with a classification head and criterion.

        Args:
            encoder (nn.Module): A ResNet-based encoder module.
            num_classes (int): Number of classes for classification.
            h_dim (int): Dimensionality of the latent feature space (default: 512).
            loss_fn (callable, optional): Loss function (default: CrossEntropyLoss).
        """
        super(ResNetClassifier, self).__init__()
        self.encoder = encoder
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.classifier = nn.Linear(h_dim, num_classes)  # Classification head

        # Define the criterion (default: CrossEntropyLoss)
        self.criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Logits for each class, shape [batch_size, num_classes].
        """
        # Extract features with encoder
        features = self.encoder(x)
        pooled = self.avgpool(features)  # Global average pooling
        pooled = pooled.view(pooled.size(0), -1)  # Flatten to [batch_size, h_dim]
        logits = self.classifier(pooled)  # Classification logits
        return logits

    def compute_loss(self, x, labels):
        """
        Compute the classification loss.

        Args:
            x (torch.Tensor): Input images.
            labels (torch.Tensor): Ground-truth labels.

        Returns:
            tuple: Total loss and logits.
        """
        logits = self.forward(x)
        loss = self.criterion(logits, labels)
        return loss, logits

    def eval_model(self, dataloader, device):
        """
        Evaluate the model on a given dataloader.

        Args:
            dataloader (DataLoader): Validation DataLoader.
            device (str): Device to run the evaluation on (e.g., 'cuda' or 'cpu').

        Returns:
            tuple: Validation loss and accuracy.
        """
        self.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, labels in dataloader:
                x, labels = x.to(device), labels.to(device)
                loss, logits = self.compute_loss(x, labels)
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = val_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy
