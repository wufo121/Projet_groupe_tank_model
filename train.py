import json
from pathlib import Path

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_PATH = "data/battle-tank-dataset"  # <-- à adapter
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 0  # Fix for macOS / Python 3.13 multiprocessing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------------------------------
# DATASETS & LOADERS
# -------------------------------------------------
train_ds = datasets.ImageFolder(Path(DATASET_PATH) / "train", transform=train_tf)
val_ds = datasets.ImageFolder(Path(DATASET_PATH) / "validation", transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

num_classes = len(train_ds.classes)
print(f"✔ Classes détectées : {num_classes}")

# -------------------------------------------------
# MODEL (Transfer Learning – ResNet18)
# -------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"📊 Epoch {epoch+1} | Train loss: {avg_train_loss:.4f} | Val acc: {val_acc:.2%}")

# -------------------------------------------------
# SAVE MODEL + CLASSES
# -------------------------------------------------
Path("artifacts").mkdir(exist_ok=True)

torch.save(model.state_dict(), "artifacts/model.pt")

with open("artifacts/classes.json", "w") as f:
    json.dump(train_ds.classes, f)

print("✅ Modèle sauvegardé dans artifacts/")