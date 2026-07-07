# JobAnalyze 6k (≈6000 parameter) Skill Classifier

A lightweight **multi-label** PyTorch model that predicts a fixed set of **skills/keywords** from a job description plus a provided **role** and **job type**.

This README documents the exact artifacts and behavior implemented in:

- `model/model.py` (training definition)
- `model/pred.py` (inference wrapper)
- `model/prep/data_prep.py` (feature creation)

Sample runs and evaluation numbers referenced from:

- `data/sample_data/test.txt`
- `data/sample_data/eval.txt`
- `data/sample_data/cli.txt`

---

## Model summary

### Task type

- **Multi-label classification** (each skill is predicted independently)

### Inputs

- `job_desc` (job description text)
- `role` (free text, appended)
- `job_type` / `type` (free text, appended)

These are concatenated during prediction as:

```text
{job_desc} {role} {job_type}
```

### Features

- TF-IDF features created by `model/prep/data_prep.py` using:
  - `TfidfVectorizer(max_features=150, stop_words='english', ngram_range=(1, 2), min_df=2)`
- Artifacts:
  - `model/prep/vectorizer.pkl`
  - `model/prep/label_vocab.json`
  - `model/prep/prepared_data.npz`

### Labels

- A fixed vocabulary of skills/keywords stored in `model/prep/label_vocab.json`
- In training artifacts:
  - `NUM_LABELS = len(VOCAB)`
  - In sample data: **48 Keywords/labels**

### Network architecture (the “6k / 6000 parameter” model)

`model/model.py` defines a small feed-forward network:

- Linear(input_dim → hidden_dim=32)
- ReLU
- Dropout(p=0.3)
- Linear(hidden_dim=32 → num_labels)

The file name and training printout refer to the total parameter count computed at runtime.

---

## Training (model/model.py)

**Do not run `model/model.py` directly for day-to-day use.** It is designed to be executed via `pipeline.py` (and/or the notebooks).

Training uses:

- Loss: `torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)`
  - `pos_weight` is computed per-label from the training set (class imbalance handling)
  - Clamped with `max=10.0`
- Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)`
- Epochs: `300`

### Outputs (saved artifacts)

At the end of training, the following are written to `model_out/`:

- `model_out/skill_classifier.pt`
- `model_out/training_history.json`

The test suite asserts these exist (see `test/test_model.py`).

---

## Inference (model/pred.py)

`model/pred.py` exposes a prediction helper:

```python
JobAnalyze_6k(job_desc, role="", job_type="", top_k=50) -> List[(str, float)]
```

Key behavior:

- Loads artifacts from repo-relative paths:
  - `model/prep/label_vocab.json`
  - `model/prep/vectorizer.pkl`
  - `model_out/skill_classifier.pt`
- Vectorizes the concatenated text with TF-IDF and produces logits through the trained network.
- Converts logits to probabilities with `sigmoid`.
- Ranks all labels by probability descending and returns the top results.

> Note: `top_k` is capped by the number of labels effectively returned from the ranked list (in the code, slicing is applied directly).

---

## Evaluation snapshot (from data/sample_data)

### Micro/Macro F1

From `data/sample_data/eval.txt`:

- **Micro-F1: 0.624**
- **Macro-F1: 0.420**

### Baseline comparison

Also from `data/sample_data/eval.txt`, baseline always predicts a fixed set of frequent labels:

- Baseline Micro-F1: **0.538**
- Baseline Macro-F1: **0.152**

The evaluation script prints:

> “Model meaningfully beats the naive baseline.”

### Per-label and “trap” analysis

The evaluation output includes per-label precision/recall/F1 and an additional heuristic:

- “trap?” flags labels where the model does not perform better than a trivial always-zero expectation (within a small tolerance).

From `data/sample_data/eval.txt`:

- Right: **15**
- Wrong: **33**
- Total labels: **48**
- Keyword Accuracy: **31.25%**

---

## Example: single inference test (data/sample_data/test.txt)

`data/sample_data/test.txt` contains a job description plus:

- Role: **AI Engineer**
- Type: **Junior**

It lists **15 keys to be predicted**, including:

- Python, LLMs, LangGraph, MCP, GenAI, VectorDB, SQL, APIs, Docker, Agents, Github, CI/CD, Git, AWS/Azure, Prompt Engineering

Reported performance:

- Accuracy (recall): **93.34%** (14/15)

---

## Example: CLI output format (data/sample_data/cli.txt)

`cli/jobauto.py` uses `JobAnalyze_6k` from `model.pred` and prints:

- The job description provided
- Role and Type
- A ranked “TOP Skills” list as `{label} {probability}` plus a text bar

From `data/sample_data/cli.txt`, the top-ranked skills include (with example probabilities):

- apis ~0.78
- langgraph ~0.78
- vectordb ~0.76
- mcp ~0.75
- langchain ~0.74
- rag ~0.72

---

## Data prep artifacts (model/prep/data_prep.py)

`model/prep/data_prep.py` creates all required inputs for training and inference.

It:

1. Loads cleaned job description dataset
2. Normalizes and fixes known skill spelling issues (`SKILLS_FIX`)
3. Applies synonym replacement using `model/prep/sym_map.py`
4. Builds a multi-hot label vector for the vocabulary
5. Vectorizes job text with TF-IDF
6. Splits into train/test and saves:
   - `prepared_data.npz`
   - `label_vocab.json`
   - `vectorizer.pkl`

---

## Reproducibility / how to use

### Requirements (high level)

See `requirements.txt` and `pyproject.toml`.

### Minimum artifacts required for prediction

For `model/pred.py` to work, these must exist:

- `model/prep/label_vocab.json`
- `model/prep/vectorizer.pkl`
- `model_out/skill_classifier.pt`

If any are missing, `model/pred.py` raises a `FileNotFoundError` with guidance.

### Recommended workflow

- Run the full pipeline via `pipeline.py` (which coordinates data prep + training).
- Use `cli/jobauto.py` for interactive predictions.

---

## Notes on “JobAnalyze 6k”

Despite the “6k / 6000 parameter” naming, the true parameter count is computed dynamically in `model/model.py`.

The implementation is intentionally small:

- TF-IDF input features
- 1 hidden layer with 32 units
- Multi-label BCE loss with class imbalance reweighting

This design keeps inference fast and model size small while still providing meaningful gains over the naive baseline (see evaluation snapshot above).
