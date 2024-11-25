import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import random
from sklearn.metrics import accuracy_score

# Paths to datasets
train_dir = "/data2/zijin/random/2024-SmallData-Project/training_dataset"
test_dir = "/data2/zijin/random/2024-SmallData-Project/testing_dataset"

# Hyperparameters
batch_size = 32
num_epochs = 150
learning_rate = 0.001

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resizing for ResNet input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
num_classes = len(train_dataset.classes)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(
    model.fc.in_features, num_classes
)  # Adjust final layer for num_classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%"
    )

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# Define the random guesser function
def random_guesser(test_loader, num_classes):
    all_labels = []
    all_predictions = []

    for _, labels in test_loader:
        batch_size = labels.size(0)
        random_predictions = [
            random.randint(0, num_classes - 1) for _ in range(batch_size)
        ]
        all_labels.extend(labels.numpy())
        all_predictions.extend(random_predictions)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy


# Calculate the baseline random accuracy
num_classes = len(test_dataset.classes)
baseline_accuracy = random_guesser(test_loader, num_classes)
print(f"Baseline Random Guesser Accuracy: {baseline_accuracy * 100:.2f}%")
