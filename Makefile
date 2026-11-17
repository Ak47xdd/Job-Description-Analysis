.PHONY: train-sbert

train-sbert:
	python pipeline.py
	python model/prep/data_prep.py
	python model/sentence_transformer_v2.py
	python model/export_sbert_classifier_numpy.py
	python model/eval_sbert.py
	python model/model_sbert.py
