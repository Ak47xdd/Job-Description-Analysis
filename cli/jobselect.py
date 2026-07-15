"""
jobselect.py — CLI entry point.
Uses relative imports so it works both as an installed pip package
(jobselect command) and during local development (python -m cli.jobselect).
"""

from rich import print

from .utils import clear_console, title, query
from .main_screen import main_screen


def cli() -> None:
    try:
        jd, role, job_type, results, mode = main_screen()

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