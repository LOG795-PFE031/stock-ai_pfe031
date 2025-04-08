"""
Progress indicator utilities using Rich.
"""
from typing import Optional, Any
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.layout import Layout
from rich.console import Group
import time

console = Console()

def create_progress_bar(
    total: int,
    description: str = "Processing...",
    show_time: bool = True
) -> Progress:
    """
    Create a progress bar with spinner and time information.
    
    Args:
        total: Total number of steps
        description: Description of the task
        show_time: Whether to show time information
        
    Returns:
        Progress instance
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ]
    
    if show_time:
        columns.extend([
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ])
    
    return Progress(*columns, console=console)

def create_spinner(description: str = "Processing...") -> Live:
    """
    Create a spinner with description.
    
    Args:
        description: Description of the task
        
    Returns:
        Live instance with spinner
    """
    spinner = Spinner("dots", description)
    return Live(spinner, console=console, refresh_per_second=10)

def create_status_panel(
    title: str,
    content: Any,
    status: str = "info"
) -> Panel:
    """
    Create a status panel with title and content.
    
    Args:
        title: Panel title
        content: Panel content
        status: Status type (info, success, warning, error)
        
    Returns:
        Panel instance
    """
    styles = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red"
    }
    
    return Panel(
        content,
        title=title,
        border_style=styles.get(status, "blue")
    )

def print_status(
    title: str,
    content: Any,
    status: str = "info",
    clear_previous: bool = True,
    delay: float = 0.5
) -> None:
    """
    Print a status message with title and content.
    
    Args:
        title: Status title
        content: Status content
        status: Status type (info, success, warning, error)
        clear_previous: Whether to clear previous output
        delay: Delay before showing status (in seconds)
    """
    if clear_previous:
        console.clear()
    
    # Add a small delay to ensure previous output is cleared
    time.sleep(delay)
    
    panel = create_status_panel(title, content, status)
    console.print(panel)

def print_error(error: Exception, clear_previous: bool = True) -> None:
    """
    Print an error message with details.
    
    Args:
        error: Exception instance
        clear_previous: Whether to clear previous output
    """
    print_status(
        "Error",
        f"{error.__class__.__name__}: {str(error)}",
        "error",
        clear_previous
    )

def create_layout() -> Layout:
    """
    Create a layout for managing multiple displays.
    
    Returns:
        Layout instance
    """
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    return layout

def update_layout(
    layout: Layout,
    header: Optional[str] = None,
    body: Optional[Any] = None,
    footer: Optional[str] = None
) -> None:
    """
    Update the layout with new content.
    
    Args:
        layout: Layout instance
        header: Header content
        body: Body content
        footer: Footer content
    """
    if header:
        layout["header"].update(Panel(header))
    if body:
        layout["body"].update(body)
    if footer:
        layout["footer"].update(Panel(footer)) 