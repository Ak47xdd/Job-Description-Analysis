import json
import re
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sym_map import SYNONYM_MAP

df = pd.read_csv(r'C:\Portfolio-Projects\Job-Description-Analysis\data\clean\cleaned_job_descriptions.csv')

SKILLS_FIX = {
    'tesnorflow/pytorch': 'tensorflow/pytorch',
    'numpyhugging face':  'numpy',
    'sytem design':       'system design',
    'python. ml':         'ml',           
}

def normalizer(skills) -> list:
    if pd.isna(skills):
        return []
    skill = [s.strip().lower() for s in skills.split(',') if s.strip()]
    return [SKILLS_FIX.get(s, s) for s in skill]

df['skill_list'] = df['tech_skills'].apply(normalizer)

freq  = Counter(s for lst in df['skill_list'] for s in lst)
VOCAB = sorted([s for s, c in freq.items() if c >= 2])

def encoder(skill_list) -> list:
    return [1 if lbl in skill_list else 0 for lbl in VOCAB]

y = np.array([encoder(lst) for lst in df['skill_list']], dtype=np.float32)

def apply_synonyms(text: str) -> str:
    text = text.lower()
    for phrase, canonical in sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(phrase, canonical)
    return text

jd_input = (
    df['job_desc'].fillna('').apply(apply_synonyms) + ' ' +
    df['role'].fillna('').str.lower() + ' ' +
    df['type'].fillna('').str.lower()
)

vectorizer = TfidfVectorizer(
    max_features=80,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
)
X = vectorizer.fit_transform(jd_input).toarray().astype(np.float32)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(df)),
    test_size=0.2,
    random_state=42
)

print(f"Dataset: {len(df)} rows | Vocab: {len(VOCAB)} labels | "
      f"TF-IDF features: {X.shape[1]}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

np.savez(
    'prepared_data.npz',
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    idx_train=idx_train,
    idx_test=idx_test
)

with open('label_vocab.json', 'w') as f:
    json.dump(VOCAB, f, indent=2)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Successfully Vectorized and Pickled Data")