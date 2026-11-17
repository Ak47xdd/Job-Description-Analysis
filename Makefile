# Use the project virtual environment so training dependencies are not looked up
# in the system-wide Python installation.
.PHONY: train-sbert

ifeq ($(OS),Windows_NT)
PYTHON := .venv/Scripts/python.exe
else
PYTHON := .venv/bin/python
endif

train-sbert:
	$(PYTHON) pipeline.py
	$(PYTHON) model/prep/data_prep.py
	$(PYTHON) model/sentence_transformer_v2.py
	$(PYTHON) model/export_sbert_classifier_numpy.py
	$(PYTHON) model/eval_sbert.py
	$(PYTHON) model/model_sbert.py
