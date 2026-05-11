import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import matplotlib.pyplot as plt

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
])

# DATASET
dataset = datasets.ImageFolder("dataset", transform=transform)

class_names = dataset.classes
print("Classes:", class_names)

# SPLIT
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DATALOADER
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# MODEL
model = timm.create_model(
    "efficientnet_b3",
    pretrained=True,
    num_classes=len(class_names)
)

model = model.to(device)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# METRICS
train_losses = []
val_accuracies = []

# TRAINING
epochs = 25

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    # VALIDATION
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Validation Accuracy: %{accuracy:.2f}\n")

# SAVE MODEL
torch.save(model.state_dict(), "breed_model.pth")

print("Model saved.")

# LOSS GRAPH
plt.figure(figsize=(8,5))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_graph.png")

# ACCURACY GRAPH
plt.figure(figsize=(8,5))
plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy_graph.png")

print("Graphs saved.")