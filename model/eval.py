import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score
from importlib import import_module
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from model import SkillClassifier

sys.path.insert(0, '.')

data = np.load('prep/prepared_data.npz')

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

with open('prep/label_vocab.json') as f:
    VOCAB = json.load(f)
    
model = SkillClassifier(X_train.shape[1], len(VOCAB))
model.load_state_dict(torch.load(ROOT / 'model_out' / 'skill_classifier.pt', map_location='cpu'))

model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X_test, dtype=torch.float32))
    probs = torch.sigmoid(logits).numpy()
    
THRESHOLD = 0.5
preds =(probs >= THRESHOLD).astype(int)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, preds, average=None, zero_division=0
)
micro_f1 = f1_score(y_test, preds, average='micro', zero_division=0)
macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)

print("\n Model Evaluation Metrics\n")
print(f"{'label':25s}{'support':10s}{'precision':12s}{'recall':10s}{'f1':6s}")
for lbl, p, r, f, s in zip(VOCAB, precision, recall, f1, support):
    if s > 0:
        print(f"{lbl:25s}{int(s):<10d}{p:<12.2f}{r:<10.2f}{f:.2f}")
        
print(f"\nMicro-F1: {micro_f1:.3f} | Macro-F1: {macro_f1:.3f}")

train_freq = y_train.mean(axis=0)  
baseline_preds = np.tile((train_freq >= 0.5).astype(int), (len(y_test), 1))

baseline_micro_f1 = f1_score(y_test, baseline_preds, average='micro', zero_division=0)
baseline_macro_f1 = f1_score(y_test, baseline_preds, average='macro', zero_division=0)

print(f"\nBASELINE: \n")
baseline_labels = [lbl for lbl, f in zip(VOCAB, train_freq) if f >= 0.5]
print(f"Baseline always predicts: {baseline_labels}")
print(f"Baseline Micro-F1: {baseline_micro_f1:.3f} | Macro-F1: {baseline_macro_f1:.3f}")

print("\nVERDICT \n")
if micro_f1 > baseline_micro_f1 + 0.05:
    print("Model meaningfully beats the naive baseline.")
else:
    print("Model is roughly tied with (or worse than) just guessing the most")
    print("common labels every time -- with only 23 training rows, this is")
    print("the expected outcome. The pipeline works; the dataset is too small")
    print("for the model to learn real text->skill patterns yet.")