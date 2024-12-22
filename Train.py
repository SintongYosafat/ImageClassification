# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from datasets import load_dataset
import wandb

# Langkah 1: Inisialisasi WandB
wandb.login()  # Pastikan Anda login dengan akun WandB
wandb.init(project="trashnet_classification", config={
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "architecture": "ResNet18"
})

# Langkah 2: Load Dataset TrashNet
ds = load_dataset("garythung/trashnet", split="train")

# Langkah 3: Preprocessing dan Transformasi Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize gambar agar sesuai dengan input model
    transforms.ToTensor(),          # Konversi ke tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisasi
])

class TrashDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]['image']
        label = self.ds[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset dengan Transformasi
dataset = TrashDataset(ds, transform=transform)

# Langkah 4: Split Dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = wandb.config["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Langkah 5: Setup Model (ResNet18 Pretrained)
model = models.resnet18(pretrained=True)
num_classes = len(set(ds['label']))
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Gunakan GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Langkah 6: Define Loss dan Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

# Langkah 7: Training Loop
num_epochs = wandb.config["epochs"]

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}")

    # Evaluation Loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    wandb.log({"epoch": epoch + 1, "validation_accuracy": accuracy})
    print(f"Validation Accuracy: {accuracy:.2f}%")

# Langkah 8: Akhiri Logging WandB
wandb.finish()

# Simpan model terlatih sebagai file
torch.save(model.state_dict(), "trained_model.pth")
