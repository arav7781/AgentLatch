from setuptools import setup

# Display the startup banner on installation directly to the controlling terminal
try:
    with open('/dev/tty', 'w') as tty:
        from rich.console import Console

        from agentlatch.banner import initialize_latch
        tty_console = Console(file=tty, force_terminal=True)
        initialize_latch(console=tty_console)
except Exception:
    pass

setup()
