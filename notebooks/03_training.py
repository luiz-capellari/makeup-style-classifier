"""
Fine-tuning do EfficientNet-B0 no dataset de maquiagem.
Exporta o modelo treinado como ONNX para inferência leve em CPU.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json

DATA_DIR   = Path("/content/processed")
MODEL_OUT  = Path("/content/models/makeup_classifier.onnx")
CLASSES = CLASSES = ["heavy", "natural", "no_makeup"]
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# --- Transforms ---
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Datasets ---
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=val_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Modelo: EfficientNet-B0 com head customizado ---
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model = model.to(DEVICE)

# Congela o backbone nas primeiras épocas (fine-tuning gradual)
for param in model.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_epoch(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# --- Loop de treino ---
best_val_acc = 0
history = []

for epoch in range(1, EPOCHS + 1):
    # Descongela backbone após época 5
    if epoch == 6:
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.1)

    train_loss, train_acc = train_epoch(model, train_dl)
    val_loss,   val_acc   = eval_epoch(model,  val_dl)
    scheduler.step()

    history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})
    print(f"Epoch {epoch:02d} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pt")
        print(f"  ✅ Melhor modelo salvo (val_acc={val_acc:.3f})")

# --- Exportação ONNX ---
model.load_state_dict(torch.load("models/best_model.pt"))
model.eval()
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
torch.onnx.export(
    model, dummy, str(MODEL_OUT),
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},
    opset_version=17,
)
print(f"✅ Modelo exportado: {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)")

# Salva mapeamento de classes
with open("models/classes.json", "w") as f:
    json.dump(train_ds.classes, f)
print("✅ Classes salvas em models/classes.json")

