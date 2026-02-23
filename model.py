import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
from copy import deepcopy

CLASSES = [
    "Cargo", "F-15", "F-16", "F-22", "F-35",
    "J-20", "Mirage", "Passenger", "Sukhoi SU-27", "Sukhoi SU-30"
]


def train(train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, output_dir: str = "checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLOv8 backbone and replace head for 10 classes
    yolo = YOLO("yolov8n-cls.pt")
    model = deepcopy(yolo.model)
    last = model.model[-1]
    in_features = last.linear.in_features
    last.linear = nn.Linear(in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_path = 0.0, os.path.join(output_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch}/{epochs} — val_acc: {val_acc:.4f}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc},
                       os.path.join(output_dir, f"epoch_{epoch:03d}.pt"))

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, best_path)
            print(f"  ⭐ Best model saved (val_acc={val_acc:.4f})")

    print(f"\nDone! Best model: {best_path}")
    return best_path
