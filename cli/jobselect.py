"""
jobselect.py — CLI entry point.
Uses relative imports so it works both as an installed pip package
(jobselect command) and during local development (python -m cli.jobselect).
"""


from rich import print
from rich.console import Console

from .api_val import val_api, infer_mode
from .utils import clear_console, title, query
from .model_select import predict


def cli() -> None:
    try:
        console = Console()

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

        with console.status("[bold green]Loading...", spinner="dots2"):
            results, mode = predict(jd, role=role, job_type=job_type)

        clear_console()
        title()

        mode_colour = "green" if mode == "API" else "cyan"
        print(f"\n [{mode_colour}][Mode: {mode}][/{mode_colour}]")

        print("\n [yellow]Job Description Provided : \n", jd.strip())
        print("\n [yellow]Job Role : \n", role)
        print("\n [yellow]Type : \n", job_type)
        print()

        print("\n [yellow]TOP Skills \n")
        for label, prob in results:
            bar = "█" * int(prob * 30)
            print(f" {label:25s} {prob:.2f}  {bar}\n")

    except KeyboardInterrupt:
        print("\n[yellow]Exiting JobSelect")
 
 
if __name__ == "__main__":
    cli()