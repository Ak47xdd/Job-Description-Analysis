import os
import subprocess
from pyfiglet import Figlet
from rich import print
from rich.console import Console

console = Console()

def clear_console() -> None:
    if os.name == "nt":
        subprocess.run("cls", shell=True)
    else:
        subprocess.run(["clear"])


def API_title() -> None:
    f = Figlet(font="slant")
    print("——————————————————————————————————————————————————————————————————————")
    print(f.renderText("JobSelect CLI"))
    print("           Copyright Akshay Babu, All rights reserved")
    print("——————————————————————————————————————————————————————————————————————")


def title() -> None:
    f = Figlet(font="slant")
    print("——————————————————————————————————————————————————————————————————————")
    print(f.renderText("JobSelect CLI"))
    print("[blue]               Running Model : JobAnalyze 6k v1.0")
    print("           Copyright Akshay Babu, All rights reserved")
    print("——————————————————————————————————————————————————————————————————————")


def query(message: str) -> None:
    print("\n[yellow][JobSelect]")
    print(f"[yellow] >> {message} : ")
    print("[You]")
    
