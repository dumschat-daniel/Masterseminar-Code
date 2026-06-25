import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# -----------------------
# Config
# -----------------------
data_path = "/home/ddms28/playground/imagenette2-320"
batch_size = 64
num_workers = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 2  
lr = 1e-4

# -----------------------
# Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# -----------------------
# Dataset
# -----------------------
train_dataset = datasets.ImageFolder(
    root=f"{data_path}/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=f"{data_path}/val",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

print("Classes:", len(train_dataset.classes))

# -----------------------
# Model
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------
# Training
# -----------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# -----------------------
# Evaluation
# -----------------------
model.eval()

correct1 = 0
correct5 = 0
total = 0

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        _, pred5 = outputs.topk(5, 1, True, True)

        total += targets.size(0)

        correct1 += (pred5[:, 0] == targets).sum().item()

        for i in range(targets.size(0)):
            if targets[i] in pred5[i]:
                correct5 += 1

print("\n===== RESULTS =====")
print("Top-1 Accuracy:", 100.0 * correct1 / total)
print("Top-5 Accuracy:", 100.0 * correct5 / total)


# -----------------------
# Save model
# -----------------------
save_path = "/home/ddms28/playground/resnet18_imagenette.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_dataset.classes,
    "accuracy_top1": 100.0 * correct1 / total,
    "accuracy_top5": 100.0 * correct5 / total,
}, save_path)

print(f"\nModel saved to: {save_path}")