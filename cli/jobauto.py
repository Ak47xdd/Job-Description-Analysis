import os
import subprocess
import sys
# import time
# import keyboard
from pathlib import Path
from rich import print
from pyfiglet import Figlet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.pred import predict

def clear_console() -> None:
    if os.name == 'nt':
        subprocess.run('cls', shell=True)
    else:
        subprocess.run(['clear'])
        
def title() -> None:
    f = Figlet(font='slant')
    print("——————————————————————————————————————————————————————————————————————")
    print(f.renderText('JobAuto CLI'))
    print("Copyright Akshay Babu, All rights reserved")
    print("——————————————————————————————————————————————————————————————————————")

def query(message) -> None:
    print("\n[yellow][JobAuto]")
    print(f"[yellow] >> {message} : ")
    print("[You]")

def cli() -> None:
    clear_console()
    
    title()
    
    print("[yellow][JobAuto] ")
    print("[yellow]>> Welcome to JobAuto!")
    
    query("Enter Job Description")
    jd = input(" >> ")
    
    query("Enter Role")
    role = input(" >> ")
    
    query("Enter Type")
    type = input(" >> ")
    
    results = predict(jd, role=role, job_type=type)
    
    clear_console()
    
    title()

    print("\n [yellow]Job Description Provided : \n", jd.strip())
    print("\n [yellow]Job Role : \n", role)
    print("\n [yellow]Type : \n", type)
    
    print()
    
    print("\n [yellow]TOP Skills \n")
    for label, prob in results:
        bar = '█' * int(prob * 30)
        print(f" {label:25s} {prob:.2f}  {bar}\n")

if __name__ == "__main__":
    cli()