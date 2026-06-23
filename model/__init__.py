"""Model package.

Exporting SkillClassifier directly from `model.py` triggers training-time
side effects (loading datasets) at import time, which breaks CLI usage.

CLI code should import from `model.pred` / `model.model` as needed.
"""



