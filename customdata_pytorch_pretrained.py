import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# CONFIGURATION
data_dir = r"C:\Users\sonu1\OneDrive\Desktop\day03\archive (5)\hymenoptera_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")


# MAIN FUNCTION

def main():
    # --- Data transforms ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- Load datasets ---
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    assert os.path.exists(train_dir), f"Train folder not found: {train_dir}"
    assert os.path.exists(val_dir), f" Val folder not found: {val_dir}"

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    print(f" Dataset loaded: {len(train_dataset)} train images, {len(val_dataset)} validation images")

    # --- Dataloaders ---
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # --- Load pretrained model ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("Loaded ResNet18 Pretrained on ImageNet")

    # --- Modify classifier for 2 classes ---
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    # --- Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAINING LOOP
    num_epochs = 1
    print("\n Starting Training...\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f" Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")

    print("\n Training complete!")

    
    # VALIDATION LOOP
   
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # SAVE MODEL
    save_path = "resnet18_custom.pth"
    torch.save(model.state_dict(), save_path)
    print(f" Model saved as {save_path}")

# SAFE ENTRY POINT (Windows)
if __name__ == "__main__":
    main()
