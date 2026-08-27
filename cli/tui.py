"""Interactive terminal UI for JobSelect.

The TUI is a presentation layer around the existing prediction functions.
It intentionally avoids the legacy interactive API validation prompts because
those prompts are incompatible with an event-driven terminal application.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Label, Select, Static, TextArea
from textual.worker import Worker, WorkerState

from .api_val import infer_mode
from .model_select import predict

ROLES = (
    ("AI Engineer", "AI Engineer"),
    ("AI Developer", "AI Developer"),
    ("Machine Learning Engineer", "Machine Learning Engineer"),
    ("Other", "Other"),
)
JOB_TYPES = (
    ("Internship", "Internship"),
    ("Junior", "Junior"),
    ("Senior", "Senior"),
)


class JobSelectTUI(App[None]):
    """Interactive, keyboard-first JobSelect terminal application."""

    TITLE = "JobSelect CLI"
    SUB_TITLE = "JobAnalyze 6k v1.0"

    CSS = """
    Screen { align: center middle; }
    #app {
        width: 92%;
        max-width: 110;
        height: 90%;
        border: round $accent;
        padding: 1 2;
    }
    #welcome { height: auto; margin-bottom: 1; }
    .label { margin-top: 1; margin-bottom: 0; text-style: bold; }
    TextArea { height: 12; margin-bottom: 1; }
    Select { width: 100%; margin-bottom: 1; }
    Button { margin: 1 1 1 0; }
    #status { height: auto; margin: 1 0; }
    #result-scroll { height: 1fr; border: round $panel; padding: 1 2; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "focus_description", "Description"),
        ("ctrl+r", "focus_role", "Role"),
        ("ctrl+t", "focus_job_type", "Type"),
        ("ctrl+a", "focus_analyze", "Analyze"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="app"):
            yield Static(
                "[bold]Welcome to JobSelect[/bold]\n"
                "Analyze job descriptions with JobAnalyze 6k.",
                id="welcome",
            )
            yield Label("Job Description", classes="label")
            yield TextArea(placeholder="Paste the job description here...", id="jd")
            yield Label("Job Role", classes="label")
            yield Select(ROLES, prompt="Select a role", id="role", allow_blank=True)
            yield Label("Job Type", classes="label")
            yield Select(JOB_TYPES, prompt="Select job type", id="job-type", allow_blank=True)
            with Horizontal():
                yield Button("Analyze", variant="success", id="analyze")
                yield Button("Clear", id="clear")
            yield Static("", id="status")
            with VerticalScroll(id="result-scroll"):
                yield Static("Results will appear here.", id="results")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#jd", TextArea).focus()
        self._update_status("Keyboard controls enabled")

    def _update_status(self, message: str) -> None:
        self.query_one("#status", Static).update(
            f"{message} | Ctrl+R: Role | Ctrl+T: Type | Ctrl+A: Analyze"
        )

    def action_focus_description(self) -> None:
        self.query_one("#jd", TextArea).focus()

    def action_focus_role(self) -> None:
        self.query_one("#role", Select).focus()
        self._update_status("Role selector focused. Enter/Space opens it, ↑/↓ chooses, Enter confirms.")

    def action_focus_job_type(self) -> None:
        self.query_one("#job-type", Select).focus()
        self._update_status("Job type selector focused. Enter/Space opens it, ↑/↓ chooses, Enter confirms.")

    def action_focus_analyze(self) -> None:
        self.query_one("#analyze", Button).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "role":
            selected = "not selected" if event.value is Select.BLANK else str(event.value)
            self._update_status(f"Role: {selected}")
        elif event.select.id == "job-type":
            selected = "not selected" if event.value is Select.BLANK else str(event.value)
            self._update_status(f"Job type: {selected}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear":
            self.query_one("#jd", TextArea).text = ""
            self.query_one("#role", Select).value = Select.BLANK
            self.query_one("#job-type", Select).value = Select.BLANK
            self.query_one("#results", Static).update("Results will appear here.")
            self._update_status("Form cleared")
            self.query_one("#jd", TextArea).focus()
            return
        if event.button.id == "analyze":
            self.start_analysis()

    def start_analysis(self) -> None:
        jd = self.query_one("#jd", TextArea).text.strip()
        role = self.query_one("#role", Select).value
        job_type = self.query_one("#job-type", Select).value
        status = self.query_one("#status", Static)

        if not jd:
            status.update("Please enter a job description.")
            self.query_one("#jd", TextArea).focus()
            return
        if role == Select.BLANK or job_type == Select.BLANK:
            status.update("Please select both a role and job type.")
            return

        self.query_one("#analyze", Button).disabled = True
        status.update("Analyzing job description... Please wait.")
        self.run_worker(
            self._run_prediction,
            jd,
            str(role),
            str(job_type),
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

        if event.state == WorkerState.SUCCESS:
            results, mode = event.worker.result
            lines = [
                f"[bold]Mode:[/bold] {mode}",
                "",
                "[bold underline]TOP SKILLS[/bold underline]",
                "",
            ]
            for label, probability in results:
                percent = probability * 100
                bar = "█" * max(0, min(30, int(probability * 30)))
                lines.append(f"{label:25} {percent:6.2f}%  {bar}")
            self.query_one("#results", Static).update("\n".join(lines))
            self._update_status("Analysis complete. You can run another analysis.")
        elif event.state == WorkerState.ERROR:
            self._update_status(f"Analysis failed: {event.worker.error}")


def run_tui() -> None:
    """Run the TUI without enabling terminal mouse reporting."""
    JobSelectTUI().run(mouse=False)
