import subprocess as sub

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

try :
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ModuleNotFoundError :
    print("Please Install nbconvert using pip or uv")
    nbformat = None
    ExecutePreprocessor = None

def run_notebook(notebook_path: str, kernel_name: str = "python3") -> None:
    if nbformat is None or ExecutePreprocessor is None:
        print(f"Skipping {notebook_path} (nbformat/nbconvert not installed)")
        return

    print(f"Starting: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)

    ep.preprocess(nb, {"metadata": {"path": "./"}})

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Finished: {notebook_path}\n")
    
def run_scripts(script_path: str, shell_name: str = "python") -> None:
    print(f"Running scripts from {script_path}")

    sub.run([f"{shell_name}", f"{script_path}"], cwd=REPO_ROOT, check=True)

run_notebook("notebooks/01_EDA.ipynb")
run_notebook("notebooks/02_Data_Engineering.ipynb")
run_scripts("model/prep/data_prep.py")
run_scripts("model/Model.py")
