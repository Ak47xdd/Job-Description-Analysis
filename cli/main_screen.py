from rich.console import Console
from rich import print

from .utils import clear_console, title, query

console = Console()

def main_screen():
    from .api_val import val_api, infer_mode

    val_api()
    infer = infer_mode()

    clear_console()
    title()

    print(f"[yellow][JobAnalyze : {infer}]")
    print("[yellow]>> Welcome to JobSelect!")

    query("Enter Job Description")
    jd = input(" >> ")

    query("Enter Role  (AI Engineer / AI Developer)")
    role = input(" >> ")

    query("Enter Type  (Internship / Junior / Senior)")
    job_type = input(" >> ")

    from .model_select import predict

    with console.status("[bold green]Loading...", spinner="dots2"):
        results, mode = predict(jd, role=role, job_type=job_type, force_local=(infer == "LOCAL"))

    return jd, role, job_type, results, mode