"""
Interactive terminal UI for JobSelect.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Button, Footer, Header, Input, Label,
    LoadingIndicator, Select, Static, TextArea,
)
from textual.worker import Worker, WorkerState
from textual.screen import Screen

from .api_val import infer_mode
from .model_select import predict


ROLES: tuple[tuple[str, str], ...] = (
    ("AI Engineer",  "AI Engineer"),
    ("AI Developer", "AI Developer"),
)
JOB_TYPES: tuple[tuple[str, str], ...] = (
    ("Internship", "Internship"),
    ("Junior",     "Junior"),
    ("Senior",     "Senior"),
)

THRESHOLD = 0.30
API_URL = "https://job-description-analysis.onrender.com"
ENV_DIR = Path.home() / ".jobselect"
ENV_FILE = ENV_DIR / ".env"


class APIKeyScreen(Screen[str]):
    """Collect and persist the API key, or allow local inference."""

    CSS = """
    APIKeyScreen { align: center middle; }
    #key-box {
        width: 80;
        max-width: 90%;
        border: round $accent;
        padding: 2 3;
        height: auto;
    }
    #key-title  { text-style: bold; margin-bottom: 1; }
    #key-hint   { color: $text-muted; margin-bottom: 1; }
    #key-input  { width: 100%; margin-bottom: 1; }
    #key-actions {
        width: 100%;
        height: auto;
        align: left middle;
    }
    #key-actions Button {
        margin: 0 1 0 0;
        height: 3;
    }
    #key-submit { min-width: 20; }
    #key-local  { min-width: 16; }
    """

    BINDINGS = [("escape", "run_local", "Run Locally")]

    def compose(self) -> ComposeResult:
        with Container(id="key-box"):
            yield Static("[bold]JobSelect Labs · JobAnalyze 6k v1.0[/bold]", id="key-title")
            yield Static(
                "Enter your API key to use cloud inference, or press [bold]Run Locally[/bold] "
                "to use the local model.\n"
                "Your key will be saved to [italic]~/.jobselect/.env[/italic] for future sessions.",
                id="key-hint",
            )
            yield Input(placeholder="ja6k_...", password=True, id="key-input")
            with Horizontal(id="key-actions"):
                yield Button("Connect →", variant="success", id="key-submit")
                yield Button("Run Locally", id="key-local")

    def on_mount(self) -> None:
        self.query_one("#key-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "key-local":
            self.action_run_local()
        elif event.button.id == "key-submit":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        entered = self.query_one("#key-input", Input).value.strip()
        if entered:
            ENV_DIR.mkdir(parents=True, exist_ok=True)
            ENV_FILE.write_text(
                f'JOBSELECT_API_URL="{API_URL}"\n'
                f'JOBSELECT_API_KEY="{entered}"\n',
                encoding="utf-8",
            )
            try:
                ENV_FILE.chmod(0o600)
            except OSError:
                pass
            os.environ["JOBSELECT_API_URL"] = API_URL
            os.environ["JOBSELECT_API_KEY"] = entered
        self.dismiss(entered)

    def action_run_local(self) -> None:
        self.dismiss("")


class JobSelectTUI(App[None]):
    TITLE     = "JobSelect CLI"
    SUB_TITLE = "JobAnalyze 6k v1.0  ·  Copyright © Akshay Babu, JobSelect Labs"

    CSS = """
    Screen { align: center middle; }
    #app {
        width: 92%;
        max-width: 120;
        height: 92%;
        border: round $accent;
        padding: 1 2;
    }
    #welcome         { height: auto; margin-bottom: 1; color: $text-muted; }
    .label           { margin-top: 1; margin-bottom: 0; text-style: bold; }
    TextArea         { height: 10; margin-bottom: 1; }
    Select           { width: 100%; margin-bottom: 1; }
    Button           { margin: 1 1 1 0; }
    #analyze         { min-width: 24; }
    #clear           { min-width: 12; }
    #status          { height: auto; margin: 1 0; color: $text-muted; }
    #loading         { height: 1; margin: 0; display: none; }
    #loading.visible { display: block; }
    #result-scroll   { height: 1fr; min-height: 8; border: round $panel; padding: 1 2; }
    #results         { width: 1fr; height: auto; }
    """

    BINDINGS = [
        ("q",      "quit",             "Quit"),
        ("escape", "focus_description", "Description"),
        ("ctrl+r", "focus_role",       "Role"),
        ("ctrl+t", "focus_job_type",   "Type"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="app"):
            yield Static(
                "Analyze job descriptions with JobAnalyze 6k. "
                "Fill the fields below and activate [bold]Analyze Job Description[/bold].",
                id="welcome",
            )
            yield Label("Job Description", classes="label")
            yield TextArea(placeholder="Paste the full job description here...", id="jd")
            yield Label("Job Role", classes="label")
            yield Select(ROLES, prompt="Select a role", id="role", allow_blank=True)
            yield Label("Job Type", classes="label")
            yield Select(JOB_TYPES, prompt="Select job type", id="job-type", allow_blank=True)
            with Horizontal():
                yield Button("Analyze Job Description", variant="success", id="analyze")
                yield Button("Clear", id="clear")
            yield Static("", id="status")
            yield LoadingIndicator(id="loading")
            with VerticalScroll(id="result-scroll"):
                yield Static(
                    "[dim]Results will appear here after analysis.[/dim]",
                    id="results",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#jd", TextArea).focus()
        self._status("Ready. Fill the fields then press Analyze Job Description.")

        # Only the API KEY determines whether the startup key screen is needed.
        # A saved API URL by itself must never suppress the key prompt.
        saved_key = os.getenv("JOBSELECT_API_KEY", "").strip()
        if not saved_key and ENV_FILE.exists():
            load_dotenv(dotenv_path=ENV_FILE, override=False)
            saved_key = os.getenv("JOBSELECT_API_KEY", "").strip()

        if not saved_key:
            self.push_screen(APIKeyScreen(), self._on_key_screen_dismissed)

    def _on_key_screen_dismissed(self, entered_key: str) -> None:
        if entered_key:
            self._status("API key saved. Running in [green]API[/green] mode.")
        else:
            self._status("Running in [cyan]LOCAL[/cyan] mode.")

    def _status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _set_loading(self, active: bool) -> None:
        loader = self.query_one("#loading", LoadingIndicator)
        if active:
            loader.add_class("visible")
        else:
            loader.remove_class("visible")

    def action_focus_description(self) -> None:
        self.query_one("#jd", TextArea).focus()

    def action_focus_role(self) -> None:
        self.query_one("#role", Select).focus()
        self._status("Role focused. Space/Enter opens, ↑↓ selects, Enter confirms.")

    def action_focus_job_type(self) -> None:
        self.query_one("#job-type", Select).focus()
        self._status("Job type focused. Space/Enter opens, ↑↓ selects, Enter confirms.")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "role":
            v = "not selected" if event.value is Select.BLANK else str(event.value)
            self._status(f"Role set to: {v}")
        elif event.select.id == "job-type":
            v = "not selected" if event.value is Select.BLANK else str(event.value)
            self._status(f"Job type set to: {v}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear":
            self._clear_form()
        elif event.button.id == "analyze":
            self.start_analysis()

    def _clear_form(self) -> None:
        self.query_one("#jd", TextArea).text = ""
        self.query_one("#role", Select).value = Select.BLANK
        self.query_one("#job-type", Select).value = Select.BLANK
        self.query_one("#results", Static).update(
            "[dim]Results will appear here after analysis.[/dim]"
        )
        self._status("Form cleared.")
        self.query_one("#jd", TextArea).focus()

    def start_analysis(self) -> None:
        jd       = self.query_one("#jd", TextArea).text.strip()
        role     = self.query_one("#role", Select).value
        job_type = self.query_one("#job-type", Select).value

        if not jd:
            self._status("[yellow]Please enter a job description.[/yellow]")
            self.query_one("#jd", TextArea).focus()
            return
        if role is Select.BLANK or job_type is Select.BLANK:
            self._status("[yellow]Please select both a job role and job type.[/yellow]")
            return

        self.query_one("#analyze", Button).disabled = True
        self._set_loading(True)
        self._status("Analyzing job description… please wait.")
        self.run_worker(
            lambda: self._run_prediction(jd, str(role), str(job_type)),
            name="job-analysis",
            exclusive=True,
            thread=True,
        )

    def _run_prediction(self, jd: str, role: str, job_type: str):
        infer = infer_mode()
        return predict(jd, role=role, job_type=job_type, force_local=(infer == "LOCAL"))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "job-analysis":
            return
        if event.state == WorkerState.RUNNING:
            return

        self.query_one("#analyze", Button).disabled = False
        self._set_loading(False)

        if event.state == WorkerState.SUCCESS:
            try:
                results, mode = event.worker.result
                self._render_results(results, mode)
                self._status("Analysis complete. You can analyze another job description.")
            except Exception as exc:
                self._status(f"[red]Could not render analysis results:[/red] {exc}")

        elif event.state == WorkerState.ERROR:
            self._status(
                f"[red]Analysis failed:[/red] {event.worker.error}\n"
                "Check your API key or switch to LOCAL mode."
            )

    def _render_results(self, results: list[tuple[str, float]], mode: str) -> None:
        jd = self.query_one("#jd", TextArea).text.strip()
        jd_snippet = jd[:120] + ("…" if len(jd) > 120 else "")

        role = str(self.query_one("#role", Select).value)
        job_type = str(self.query_one("#job-type", Select).value)
        mode_markup = "[green]API[/green]" if mode == "API" else "[cyan]LOCAL[/cyan]"

        lines = [
            "[bold underline]JOB DESCRIPTION PROVIDED[/bold underline]",
            escape(jd_snippet),
            "",
            f"[bold]Role:[/bold]  {escape(role)}",
            f"[bold]Type:[/bold]  {escape(job_type)}",
            f"[bold]Mode:[/bold]  {mode_markup}",
            "",
            "[bold underline]TOP SKILLS[/bold underline]",
            "",
        ]

        above_threshold = [
            (str(label), float(prob))
            for label, prob in results
            if float(prob) >= THRESHOLD
        ]

        if not above_threshold:
            lines.append(
                "[dim]No skills predicted above threshold (0.30). "
                "Showing the highest-scoring skills instead:[/dim]"
            )
            above_threshold = [
                (str(label), float(prob)) for label, prob in list(results)[:10]
            ]

        for label, prob in above_threshold[:20]:
            bar = "█" * max(0, min(30, int(prob * 30)))
            percent = prob * 100
            lines.append(f"{escape(label):25} {percent:5.1f}%  {bar}")

        self.query_one("#results", Static).update("\n".join(lines))


def run_tui() -> None:
    JobSelectTUI().run(mouse=False)
