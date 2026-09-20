
from __future__ import annotations
import gc, json, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = max(64, int(os.getenv("SBERT_MAX_SEQ_LENGTH", "128")))
BATCH_SIZE = max(1, int(os.getenv("SBERT_BATCH_SIZE", "1")))
SBERT_BACKEND = os.getenv("SBERT_BACKEND", "onnx").strip().lower()
REQUIRE_ONNX = os.getenv("SBERT_REQUIRE_ONNX", "true").strip().lower() not in {"0","false","no"}
SBERT_ONNX_FILE = os.getenv("SBERT_ONNX_FILE", "onnx/model.onnx").strip()
SBERT_ONNX_PROVIDER = os.getenv("SBERT_ONNX_PROVIDER", "CPUExecutionProvider").strip()
SBERT_ONNX_DISABLE_CPU_ARENA = os.getenv("SBERT_ONNX_DISABLE_CPU_ARENA", "true").strip().lower() not in {"0","false","no"}

_tokenizer = None
_onnx_session = None
_classifier_weights = None
_label_vocab = None
_embedding_dim = None

def _normalise_model_repo(model_name):
    if "/" not in model_name and not Path(model_name).exists():
        return f"sentence-transformers/{model_name}"
    return model_name

def _resolve_onnx_path(model_name):
    model_path = Path(model_name)
    if model_path.exists():
        candidate = model_path / SBERT_ONNX_FILE
        if not candidate.exists():
            raise FileNotFoundError(f"Missing ONNX model {candidate}. Expected SBERT_ONNX_FILE={SBERT_ONNX_FILE}.")
        return str(candidate)
    try:
        return hf_hub_download(repo_id=_normalise_model_repo(model_name), filename=SBERT_ONNX_FILE)
    except Exception as exc:
        raise RuntimeError(f"Could not download ONNX model '{SBERT_ONNX_FILE}'.") from exc

def _load_embedding_runtime(model_name):
    global _tokenizer, _onnx_session, _embedding_dim
    if REQUIRE_ONNX and SBERT_BACKEND != "onnx":
        raise RuntimeError("Render-safe SBERT requires SBERT_BACKEND=onnx.")
    if _tokenizer is not None and _onnx_session is not None and _embedding_dim is not None:
        return _tokenizer, _onnx_session, _embedding_dim
    tokenizer = AutoTokenizer.from_pretrained(_normalise_model_repo(model_name), use_fast=True)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_mem_pattern = False
    session_options.enable_mem_reuse = True
    session_options.enable_cpu_mem_arena = not SBERT_ONNX_DISABLE_CPU_ARENA
    session = ort.InferenceSession(_resolve_onnx_path(model_name), sess_options=session_options, providers=[SBERT_ONNX_PROVIDER])
    outputs = session.get_outputs()
    if not outputs or len(outputs[0].shape) != 3:
        raise RuntimeError("Expected ONNX transformer output [batch, sequence, hidden].")
    hidden_dim = outputs[0].shape[-1]
    if not isinstance(hidden_dim, int):
        raise RuntimeError(f"Could not determine embedding dimension from {outputs[0].shape}.")
    _tokenizer, _onnx_session, _embedding_dim = tokenizer, session, hidden_dim
    return tokenizer, session, hidden_dim

def _load_numpy_classifier(weights_path, input_dim, num_labels, hidden_dim):
    npz_path = weights_path.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing lightweight classifier artifact: {npz_path}. "
            "Run python model/export_sbert_classifier_numpy.py locally and commit the generated .npz."
        )
    with np.load(npz_path, allow_pickle=False) as data:
        w1 = np.asarray(data["w1"], dtype=np.float32)
        b1 = np.asarray(data["b1"], dtype=np.float32)
        w2 = np.asarray(data["w2"], dtype=np.float32)
        b2 = np.asarray(data["b2"], dtype=np.float32)
    expected = [(w1,(hidden_dim,input_dim)),(b1,(hidden_dim,)),(w2,(num_labels,hidden_dim)),(b2,(num_labels,))]
    for value, shape in expected:
        if value.shape != shape:
            raise ValueError(f"Invalid NumPy classifier shape {value.shape}; expected {shape}.")
    return w1,b1,w2,b2

def _load_artifacts():
    global _classifier_weights, _label_vocab
    if _tokenizer is not None and _onnx_session is not None and _classifier_weights is not None and _label_vocab is not None and _embedding_dim is not None:
        return _tokenizer,_onnx_session,_classifier_weights,_label_vocab,_embedding_dim
    label_path = ROOT/"model"/"prep"/"v2"/"label_vocab_v2.json"
    out_dir = ROOT/"model_out"/"v2_sentence_transformer"
    weights_path = out_dir/"skill_classifier_sbert_v2.pt"
    config_path = out_dir/"model_config.json"
    if not label_path.exists(): raise FileNotFoundError(f"Missing {label_path}.")
    if not weights_path.exists(): raise FileNotFoundError(f"Missing {weights_path}.")
    with label_path.open(encoding="utf-8") as f: label_vocab=json.load(f)
    if not isinstance(label_vocab,list) or not label_vocab or len(label_vocab)!=len(set(label_vocab)):
        raise ValueError(f"Invalid or duplicate SBERT label vocabulary: {label_path}")
    config={}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f: config=json.load(f)
    if config.get("num_labels") is not None and int(config["num_labels"]) != len(label_vocab):
        raise ValueError("SBERT config and vocabulary label counts disagree.")
    input_dim=int(config.get("input_dim",384))
    hidden_dim=int(config.get("hidden_dim",64))
    tokenizer,session,embedding_dim=_load_embedding_runtime(config.get("embedding_model",DEFAULT_EMBEDDING_MODEL))
    if embedding_dim != input_dim:
        raise ValueError(f"SBERT embedding dimension mismatch: ONNX encoder produces {embedding_dim}, classifier expects {input_dim}.")
    weights=_load_numpy_classifier(weights_path,input_dim,len(label_vocab),hidden_dim)
    _classifier_weights,_label_vocab=weights,label_vocab
    return tokenizer,session,weights,label_vocab,embedding_dim

def _build_input_text(job_desc, role, job_type):
    return f"Role: {role}. Job type: {job_type}. Job description: {job_desc}"

def _encode_embeddings(texts, tokenizer, session):
    encoded=tokenizer(texts,padding=True,truncation=True,max_length=MAX_SEQ_LENGTH,return_tensors="np")
    session_inputs={item.name for item in session.get_inputs()}
    ort_inputs={name:np.asarray(value,dtype=np.int64) for name,value in encoded.items() if name in session_inputs}
    missing=session_inputs.difference(ort_inputs)
    if missing: raise RuntimeError(f"Tokenizer did not produce required ONNX inputs: {sorted(missing)}")
    token_embeddings=session.run(None,ort_inputs)[0].astype(np.float32,copy=False)
    mask=encoded["attention_mask"].astype(np.float32,copy=False)[:,:,None]
    summed=np.sum(token_embeddings*mask,axis=1,dtype=np.float32)
    counts=np.clip(mask.sum(axis=1,dtype=np.float32),1e-9,None)
    embeddings=summed/counts
    norms=np.linalg.norm(embeddings,axis=1,keepdims=True)
    return np.asarray(embeddings/np.clip(norms,1e-12,None),dtype=np.float32)

def _classifier_predict(embeddings,weights):
    w1,b1,w2,b2=weights
    hidden=np.maximum(0.0,embeddings@w1.T+b1)
    logits=np.clip(hidden@w2.T+b2,-60.0,60.0)
    return (1.0/(1.0+np.exp(-logits))).astype(np.float32,copy=False)

def JobAnalyze_v2_SBERT(job_desc="",role="",job_type="",top_k=50):
    return JobAnalyze_v2_SBERT_batch([job_desc],role,job_type,top_k)[0]

def JobAnalyze_v2_SBERT_batch(job_descs,role="",job_type="",top_k=50):
    if top_k<1: return [[] for _ in job_descs]
    if not job_descs: return []
    tokenizer,session,weights,label_vocab,_=_load_artifacts()
    texts=[_build_input_text(text or "",role,job_type) for text in job_descs]
    results=[]
    for start in range(0,len(texts),BATCH_SIZE):
        embeddings=_encode_embeddings(texts[start:start+BATCH_SIZE],tokenizer,session)
        probabilities=_classifier_predict(embeddings,weights)
        for row in probabilities:
            ranked=sorted(zip(label_vocab,row),key=lambda item:-float(item[1]))
            results.append([(skill,float(score)) for skill,score in ranked[:top_k]])
        del embeddings,probabilities
        gc.collect()
    return results
