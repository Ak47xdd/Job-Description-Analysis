""" 
model.py - DO NOT RUN THIS SCRIPT!

Run this script, prep/data_prep.py and the notebooks through pipeline.py
"""

import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

torch.manual_seed(42)

DATA_DIR = Path(__file__).resolve().parent / 'prep'

# Ensure required artifacts exist even when run from different CWDs
assert (DATA_DIR / 'prepared_data.npz').exists(), f"Missing {DATA_DIR / 'prepared_data.npz'}"
assert (DATA_DIR / 'label_vocab.json').exists(), f"Missing {DATA_DIR / 'label_vocab.json'}"

data = np.load(DATA_DIR / 'prepared_data.npz')

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

with open(DATA_DIR / 'label_vocab.json', encoding='utf-8') as f:
    VOCAB = json.load(f)

    
DIM = X_train.shape[1]
NUM_LABELS = len(VOCAB)

class SkillData(Dataset):
    def __init__(self, X, y) -> torch.tensor:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self) -> len:
        return len(self.X)
    
    def __getitem__(self, idx) -> int:
        return self.X[idx], self.y[idx]
    
train_ds = SkillData(X_train, y_train)
test_ds = SkillData(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

class SkillClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=32, dropout=0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)       
        )
        
    def forward(self, x) -> float:
        return self.net(x)
    
model = SkillClassifier(DIM, NUM_LABELS)

pos_counts = y_train.sum(axis=0)
neg_counts = len(y_train) - pos_counts
pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)
pos_weight = torch.clamp(pos_weight, max=10.0)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 380
history = {'train_loss' : [], 'test_loss' : []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
    model.eval()
    test_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            test_losses.append(loss.item())
            
    train_loss = sum(train_losses) / len(train_losses)
    test_loss = sum(test_losses) / len(test_losses)
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | test_loss {test_loss:.4f}")
        
out_dir = REPO_ROOT / 'model_out'
out_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), out_dir / 'skill_classifier.pt')
with open(out_dir / 'training_history.json', 'w') as f:
    json.dump(history, f)       
    
print("\n Saved Model")
print(f"Final train_loss : {history['train_loss'][-1]:.4f} | "
      f"Final test_loss : {history['test_loss'][-1]:.4f}")