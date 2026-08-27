"""JobSelect CLI entry point."""

from .tui import run_tui


def cli() -> None:
    """Start the interactive terminal UI."""
    try:
        run_tui()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
