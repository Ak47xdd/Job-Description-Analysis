import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from importlib import import_module
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from model import SkillClassifier

sys.path.insert(0, '.')

data = np.load('prep/prepared_data.npz')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

with open('prep/label_vocab.json') as f:
    VOCAB = json.load(f)
    
model = SkillClassifier(X_train.shape[1], len(VOCAB))
model.load_state_dict(torch.load(ROOT / 'model_out' / 'skill_classifier.pt', map_location='cpu'))

model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X_test, dtype=torch.float32))
    probs = torch.sigmoid(logits).numpy()
    
THRESHOLD = 0.5
preds = (probs >= THRESHOLD).astype(int)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    preds,
    average=None,
    zero_division=0
)
micro_f1 = f1_score(y_test, preds, average='micro', zero_division=0)
macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)

print("\n Model Evaluation Metrics\n")
print(f"{'label':25s}{'support':10s}{'precision':12s}{'recall':10s}{'f1':6s}")
for lbl, p, r, f, s in zip(VOCAB, precision, recall, f1, support):
    if s > 0:
        print(f"{lbl:25s}{int(s):<10d}{p:<12.2f}{r:<10.2f}{f:.2f}")
        
print(f"\nMicro-F1: {micro_f1:.3f} | Macro-F1: {macro_f1:.3f}")

print("\n=== PER-LABEL ACCURACY ===")
print(f"{'label':25s}{'accuracy%':12s}{'support':10s}{'trap?':6s}")

is_right = 0
is_wrong = 0

for i, lbl in enumerate(VOCAB):
    label_acc = accuracy_score(y_test[:, i], preds[:, i])
    s = int(support[i])
    
    always_zero_acc = 1.0 - (y_test[:, i].sum() / len(y_test))
    is_trap = always_zero_acc >= label_acc - 0.01
    
    trap_flag = "Wrong" if is_trap else "Right"
    print(f"{lbl:25s}{label_acc*100:<12.1f}{s:<10d}{trap_flag}")

    if trap_flag == 'Right':
        is_right += 1
    else:
        is_wrong += 1
        
total_labels = is_right + is_wrong

print("Right : \n", is_right)
print("Wrong : \n", is_wrong)

print("Total Labels : \n", total_labels)

key_acc = (is_right / total_labels) * 100

print(f"Keyword Accuracy : {round(key_acc, 2)}%\n")

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
