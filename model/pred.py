import json
import pickle
import numpy as np
import torch
import torch.nn
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from model import SkillClassifier

def predict(job_desc, role="", job_type="", top_k=20):
    with open('prep/label_vocab.json') as f:
        label_vocab = json.load(f)
    with open('prep/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    model = SkillClassifier(80, len(label_vocab))    
    model.load_state_dict(torch.load(ROOT / 'model_out' / 'skill_classifier.pt', map_location='cpu'))
    
    model.eval()
    
    combined_text = f"{job_desc} {role} {job_type}"
    X = vectorizer.transform([combined_text]).toarray().astype(np.float32)
    
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.sigmoid(logits).numpy()[0]
        
    ranked = sorted(zip(label_vocab, probs), key=lambda x: -x[1])
    return ranked[:top_k]
    
if __name__ == "__main__":
    jd = """
    We're looking for someone comfortable with Python and building
    chatbots powered by large language models. Experience with LangChain
    or similar agent frameworks is a plus. You should know how to work
    with vector databases for retrieval-augmented generation and be
    comfortable calling REST APIs and Docker, Knowledge in Agents is a must.
    """
    
    results = predict(jd, role="AI Engineer", job_type="Internship")
    
    print("\n Job Description Provided : \n")
    print(jd.strip())
    print("\n TOP Skills \n")
    for label, prob in results:
        bar = '#' * int(prob * 30)
        print(f" {label:25s} {prob:.2f}  {bar}")
        