"""Interactive terminal UI for JobSelect.

The TUI is a presentation layer around the existing prediction functions.
It intentionally avoids the legacy interactive API validation prompts because
those prompts are incompatible with an event-driven terminal application.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Label, Select, Static, TextArea

from .api_val import infer_mode
from .model_select import predict


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
            yield TextArea(
                placeholder="Paste the job description here...",
                id="jd",
            )
            yield Label("Job Role", classes="label")
            yield Select(
                [
                    ("AI Engineer", "AI Engineer"),
                    ("AI Developer", "AI Developer"),
                    ("Machine Learning Engineer", "Machine Learning Engineer"),
                    ("Other", "Other"),
                ],
                prompt="Select a role",
                id="role",
            )
            yield Label("Job Type", classes="label")
            yield Select(
                [
                    ("Internship", "Internship"),
                    ("Junior", "Junior"),
                    ("Senior", "Senior"),
                ],
                prompt="Select job type",
                id="job-type",
            )
            with Horizontal():
                yield Button("Analyze", variant="success", id="analyze")
                yield Button("Clear", id="clear")
            yield Static("", id="status")
            with VerticalScroll(id="result-scroll"):
                yield Static("Results will appear here.", id="results")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#jd", TextArea).focus()
        self.query_one("#status", Static).update(
            f"Mode: {infer_mode()} | Keyboard controls only"
        )

    def action_focus_description(self) -> None:
        self.query_one("#jd", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear":
            self.query_one("#jd", TextArea).text = ""
            self.query_one("#role", Select).value = Select.BLANK
            self.query_one("#job-type", Select).value = Select.BLANK
            self.query_one("#results", Static).update("Results will appear here.")
            self.query_one("#status", Static).update(
                f"Mode: {infer_mode()} | Keyboard controls only"
            )
            self.query_one("#jd", TextArea).focus()
            return

        if event.button.id != "analyze":
            return

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

        try:
            status.update("Analyzing...")
            infer = infer_mode()
            results, mode = predict(
                jd,
                role=str(role),
                job_type=str(job_type),
                force_local=(infer == "LOCAL"),
            )

            lines = [
                f"[bold]Mode:[/bold] {mode}",
                f"[bold]Role:[/bold] {role}",
                f"[bold]Type:[/bold] {job_type}",
                "",
                "[bold underline]TOP SKILLS[/bold underline]",
                "",
            ]
            for label, probability in results:
                percent = probability * 100
                bar = "█" * max(0, min(30, int(probability * 30)))
                lines.append(f"{label:25} {percent:6.2f}%  {bar}")

            self.query_one("#results", Static).update("\n".join(lines))
            status.update("Analysis complete. Press Q to exit or run another analysis.")
        except Exception as exc:
            status.update(f"Analysis failed: {exc}")


def run_tui() -> None:
    """Run the TUI without enabling terminal mouse reporting."""
    JobSelectTUI().run(mouse=False)
