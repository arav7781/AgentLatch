"""Terminal flamegraph renderer using the Rich library.

Draws a color-coded execution timeline directly in the CLI — no dashboards,
no API keys, no cloud services.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentlatch._types import EventStatus
from agentlatch.tracker import TraceEvent

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------

_COLORS = {
    EventStatus.SUCCESS: "bright_green",
    EventStatus.ERROR: "bright_red",
    EventStatus.TIMEOUT: "bright_yellow",
    EventStatus.RETRY: "bright_magenta",
    EventStatus.MEMORY_OP: "bright_cyan",
    EventStatus.LEARNING: "bright_yellow",
    "llm": "bright_blue",
    "dim": "dim",
    "header": "bold bright_cyan",
    "label": "white",
}

_BAR_CHARS = {
    EventStatus.SUCCESS: "█",
    EventStatus.ERROR: "█",
    EventStatus.TIMEOUT: "▒",
    EventStatus.RETRY: "▓",
    EventStatus.MEMORY_OP: "░",
    EventStatus.LEARNING: "▒",
    "llm": "░",
    "gap": "░",
}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Human-friendly duration string."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.0f}ms"
    return f"{seconds:.2f}s"


def _build_timeline_bar(
    trace: TraceEvent,
    bar_width: int = 60,
) -> list[tuple[Text, str]]:
    """Build colored bar segments for each child event.

    Returns a list of (bar_line, label_line) tuples — one per row.
    """
    total = trace.duration
    if total <= 0 or not trace.children:
        return []

    rows: list[tuple[Text, str]] = []

    # Sort children by start_time for correct positioning.
    sorted_children = sorted(trace.children, key=lambda c: c.start_time)

    bar = Text()
    label_parts: list[str] = []

    cursor = trace.start_time  # absolute position along the timeline

    for child in sorted_children:
        # Gap before this child = LLM reasoning time.
        gap = child.start_time - cursor
        if gap > 0:
            gap_cols = max(1, int((gap / total) * bar_width))
            bar.append(_BAR_CHARS["gap"] * gap_cols, style=_COLORS["llm"])

        # The child's own bar segment.
        child_dur = child.duration
        child_cols = max(1, int((child_dur / total) * bar_width))
        color = _COLORS.get(child.status, "white")
        bar.append(_BAR_CHARS.get(child.status, "█") * child_cols, style=color)

        status_tag = ""
        if child.status == EventStatus.ERROR:
            status_tag = " ✗ ERROR"
        elif child.status == EventStatus.TIMEOUT:
            status_tag = " ⏱ TIMEOUT"
        elif child.status == EventStatus.RETRY:
            status_tag = " ↻ RETRY"
        elif child.status == EventStatus.MEMORY_OP:
            status_tag = " 🧠 MEMORY"
        elif child.status == EventStatus.LEARNING:
            status_tag = " 📖 LEARNING"

        label_parts.append(f"{child.name} {_format_duration(child_dur)}{status_tag}")

        cursor = child.start_time + child_dur

    # Trailing gap (LLM reasoning after last tool).
    trailing = (trace.start_time + total) - cursor
    if trailing > 0:
        trail_cols = max(1, int((trailing / total) * bar_width))
        bar.append(_BAR_CHARS["gap"] * trail_cols, style=_COLORS["llm"])

    label_line = "  │  ".join(label_parts)
    rows.append((bar, label_line))

    return rows


def _compute_tool_time(trace: TraceEvent) -> float:
    """Sum of all direct children durations."""
    return sum(c.duration for c in trace.children)


def _build_summary_table(trace: TraceEvent) -> Table:
    """Rich table with per-tool breakdown."""
    table = Table(
        show_header=True,
        header_style="bold bright_white",
        border_style="dim",
        pad_edge=True,
        expand=True,
    )
    table.add_column("Tool", style="bright_white", ratio=3)
    table.add_column("Duration", style="bright_cyan", justify="right", ratio=1)
    table.add_column("Status", justify="center", ratio=1)
    table.add_column("Error", style="dim", ratio=3)

    for child in trace.children:
        status_style = _COLORS.get(child.status, "white")
        status_text = Text(child.status.value.upper(), style=status_style)

        error_info = ""
        if child.error_payload:
            error_info = (
                f"{child.error_payload.get('error_type', '?')}: "
                f"{child.error_payload.get('message', '')}"
            )

        table.add_row(
            child.name,
            _format_duration(child.duration),
            status_text,
            error_info,
        )

    return table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_flamegraph(
    trace: TraceEvent,
    console: Console | None = None,
) -> None:
    """Render a color-coded execution flamegraph to the terminal.

    Args:
        trace:    The finalized root ``TraceEvent`` tree.
        console:  Optional Rich Console (a new one is created if omitted).
    """
    con = console or Console()

    total = trace.duration
    tool_time = _compute_tool_time(trace)
    llm_time = max(0.0, total - tool_time)

    # Count memory and retry events.
    memory_ops = sum(1 for c in trace.children if c.status == EventStatus.MEMORY_OP)
    retries = sum(1 for c in trace.children if c.status == EventStatus.RETRY)
    learnings = sum(1 for c in trace.children if c.status == EventStatus.LEARNING)

    # -- Header ----------------------------------------------------------
    header = Text()
    header.append("⚡ AGENTLATCH EXECUTION PROFILE\n", style=_COLORS["header"])
    header.append(
        f"   Total: {_format_duration(total)}  │  "
        f"Tools: {_format_duration(tool_time)}  │  "
        f"LLM Reasoning: {_format_duration(llm_time)}",
        style="bright_white",
    )
    if memory_ops or retries or learnings:
        header.append("\n", style="bright_white")
        parts = []
        if memory_ops:
            parts.append(f"Memory: {memory_ops}")
        if retries:
            parts.append(f"Retries: {retries}")
        if learnings:
            parts.append(f"Learnings: {learnings}")
        header.append("   " + "  │  ".join(parts), style="bright_white")

    con.print()
    con.print(Panel(header, border_style="bright_cyan", padding=(0, 2)))

    if not trace.children:
        con.print("  [dim]No tool calls recorded.[/dim]\n")
        return

    # -- Flamegraph bars -------------------------------------------------
    rows = _build_timeline_bar(trace)
    con.print()

    # Root bar.
    root_bar = Text("█" * 60, style="bright_cyan")
    con.print("  ", end="")
    con.print(root_bar)
    con.print(
        f"  [bright_white]{trace.name}[/bright_white]"
        f"  [dim]{_format_duration(total)}[/dim]"
    )
    con.print()

    for bar, label in rows:
        con.print("  ", end="")
        con.print(bar)
        con.print(f"  [dim]{label}[/dim]")

    con.print()

    # -- Summary table ---------------------------------------------------
    table = _build_summary_table(trace)
    con.print(table)

    # -- Legend ----------------------------------------------------------
    legend = Text()
    legend.append("  Legend: ", style="dim")
    legend.append("█ ", style=_COLORS["llm"])
    legend.append("LLM Reasoning  ", style="dim")
    legend.append("█ ", style=_COLORS[EventStatus.SUCCESS])
    legend.append("Tool (OK)  ", style="dim")
    legend.append("█ ", style=_COLORS[EventStatus.ERROR])
    legend.append("Tool (ERROR)  ", style="dim")
    legend.append("▒ ", style=_COLORS[EventStatus.TIMEOUT])
    legend.append("Tool (TIMEOUT)  ", style="dim")
    legend.append("▓ ", style=_COLORS[EventStatus.RETRY])
    legend.append("Retry  ", style="dim")
    legend.append("░ ", style=_COLORS[EventStatus.MEMORY_OP])
    legend.append("Memory  ", style="dim")
    legend.append("▒ ", style=_COLORS[EventStatus.LEARNING])
    legend.append("Learning", style="dim")

    con.print(legend)
    con.print()
