from cli.utils import *
from rich import print
from cli.model_select import *

def cli() -> None:
    try:
        clear_console()
        API_title()
        query("Enter API Key (Press Enter if None : Runs Locally)")
        key = input(" >> ")
        
        if key or API_KEY:
            infer = "API"
        else :
            infer = "LOCAL"
        
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
        print("Exiting JobSelect")


if __name__ == "__main__":
    cli()
