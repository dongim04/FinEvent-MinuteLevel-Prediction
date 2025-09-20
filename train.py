import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Dataset
# -------------------
class FinAudioTextDataset(Dataset):
    def __init__(self, parquet_file):
        df = pd.read_parquet(parquet_file)
        cleaned_embs = []
        for emb in df['embeddings']:
            arr = np.array(emb, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e5, neginf=-1e5)
            cleaned_embs.append(arr)
        
        # Standardize embeddings and save scaler
        self.scaler = StandardScaler()
        all_embs = self.scaler.fit_transform(np.vstack(cleaned_embs))
        self.embeddings = [all_embs[i] for i in range(len(all_embs))]
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return emb, label


# -------------------
# Load and split dataset
# -------------------
df = pd.read_parquet("datasets/processed/final_dataset.parquet")
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]
df_2 = df[df['label'] == 2]
min_count = min(len(df_0), len(df_1))

# Downsample label 2
df_2_downsampled = df_2.sample(n=min_count, random_state=42)
df_balanced = pd.concat([df_0, df_1, df_2_downsampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced.to_parquet("datasets/processed/final_dataset_balanced.parquet")

full_dataset = FinAudioTextDataset("datasets/processed/final_dataset_balanced.parquet")

n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# -------------------
# Classifier Model
# -------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


sample_emb, _ = next(iter(train_loader))
input_dim = sample_emb.shape[1]

model = MLPClassifier(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
GRAD_CLIP = 1.0

# -------------------
# Training Loop with Validation + Early Stopping + Scheduler
# -------------------
EPOCHS = 50
PATIENCE = 7
best_val_loss = float("inf")
patience_counter = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss, correct, total = 0, 0, 0
    for embs, labels in train_loader:
        embs, labels = embs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for embs, labels in val_loader:
            embs, labels = embs.to(device), labels.to(device)
            outputs = model(embs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    # Save history
    history["train_loss"].append(total_loss/len(train_loader))
    history["val_loss"].append(val_loss/len(val_loader))
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {val_acc:.4f}, "
          f"LR: {history['lr'][-1]:.6f}")

    # Scheduler step
    scheduler.step(history["val_loss"][-1])

    # Early stopping
    if history["val_loss"][-1] < best_val_loss:
        best_val_loss = history["val_loss"][-1]
        patience_counter = 0
        torch.save(model.state_dict(), "models/mlp_classifier_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save final model + history
torch.save(model.state_dict(), "models/mlp_classifier_last.pth")
pd.DataFrame(history).to_csv("models/training_history.csv", index=False)

# -------------------
# Save for deployment (Pickle files)
# -------------------
# Load best weights first
model.load_state_dict(torch.load("models/mlp_classifier_best.pth"))
model.eval()

# Save the scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(full_dataset.scaler, f)
print("Scaler saved to models/scaler.pkl")

# Save model configuration for reconstruction
model_config = {
    'input_dim': input_dim,
    'hidden_dim': 256,
    'num_classes': 3,
    'state_dict': model.state_dict()
}

with open("models/model_config.pkl", "wb") as f:
    pickle.dump(model_config, f)
print("Model configuration saved to models/model_config.pkl")

# Alternative: Save the entire model (less recommended but simpler)
# Move model to CPU for pickle compatibility
model_cpu = model.cpu()
with open("models/full_model.pkl", "wb") as f:
    pickle.dump(model_cpu, f)
print("Full model saved to models/full_model.pkl")

# Example input for tracing (keeping original TorchScript approach)
sample_emb, _ = next(iter(train_loader))
sample_emb = sample_emb[0:1].to(device)

# Script the model (TorchScript)
traced_model = torch.jit.trace(model.to(device), sample_emb)
traced_model.save("models/mlp_classifier_deploy.pt")
print("Deployment-ready model saved to models/mlp_classifier_deploy.pt")

# -------------------
# Final Test Evaluation
# -------------------
model.load_state_dict(torch.load("models/mlp_classifier_best.pth"))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for embs, labels in test_loader:
        embs, labels = embs.to(device), labels.to(device)
        outputs = model(embs)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f"Final Test Accuracy: {test_acc:.4f}")

# -------------------
# Save label mapping for deployment
# -------------------
# Assuming you have label names - adjust as needed
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}  # Modify based on your actual labels
with open("models/label_mapping.pkl", "wb") as f:
    pickle.dump(label_mapping, f)
print("Label mapping saved to models/label_mapping.pkl")

# -------------------
# Plot training curves (save only)
# -------------------
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,3,2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,3,3)
plt.plot(history["lr"], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.legend()

plt.tight_layout()
plt.savefig("models/training_curves.png")
plt.close()
print("Training curves saved to models/training_curves.png")

print("\n=== Files saved for Streamlit deployment ===")
print("- models/scaler.pkl (StandardScaler)")
print("- models/model_config.pkl (Model configuration + weights)")
print("- models/full_model.pkl (Complete model)")
print("- models/label_mapping.pkl (Label names)")
print("- models/mlp_classifier_deploy.pt (TorchScript version)")