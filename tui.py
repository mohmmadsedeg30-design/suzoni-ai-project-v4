#!/usr/bin/env python3
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich.table import Table
from chat import SuzoniChat

console = Console()

class SuzoniTUI:
    def __init__(self):
        try:
            # نحاول تحميل النموذج النهائي، وإذا لم يوجد نستخدم الأولي
            import os
            model_path = "checkpoints/suzoni_final.pt"
            if not os.path.exists(model_path):
                model_path = "checkpoints/suzoni_initial.pt"
            
            self.chat_engine = SuzoniChat(mp=model_path)
            self.personality = self.chat_engine.c.PERSONALITY
        except Exception as e:
            console.print(f"[bold red]Error loading Suzoni AI:[/bold red] {e}")
            sys.exit(1)

    def display_welcome(self):
        welcome_text = Text()
        welcome_text.append(f"Welcome to ", style="bold white")
        welcome_text.append(f"Suzoni AI", style="bold cyan")
        welcome_text.append(f" v{self.chat_engine.c.VERSION}\n", style="italic magenta")
        
        info_table = Table(show_header=False, box=None)
        info_table.add_row("Name:", f"[cyan]{self.personality['name']}[/cyan]")
        info_table.add_row("Style:", f"[yellow]{self.personality['speaking_style']}[/yellow]")
        info_table.add_row("Traits:", f"[green]{', '.join(self.personality['traits'])}[/green]")
        
        console.print(Panel(welcome_text, subtitle="The Next Gen AI", border_style="cyan"))
        console.print(info_table)
        console.print("\n[dim]Commands: /exit, /clear, /save, /config[/dim]\n")

    def run(self):
        self.display_welcome()
        
        while True:
            user_input = Prompt.ask(f"[bold green]You[/bold green]")
            
            if user_input.lower() in ["/exit", "quit", "exit"]:
                console.print(f"\n[bold cyan]{self.personality['name']}:[/bold cyan] {self.personality['catchphrase']}! Goodbye! 👋")
                break
            
            if user_input.lower() == "/clear":
                self.chat_engine.h = []
                console.print("[italic yellow]Conversation cleared.[/italic yellow]")
                continue

            if user_input.lower() == "/config":
                console.print(Panel(f"Device: {self.chat_engine.c.DEVICE}\nModel: {self.chat_engine.c.NAME}\nHistory: {len(self.chat_engine.h)} messages", title="Configuration"))
                continue

            with console.status(f"[bold cyan]{self.personality['name']} is thinking...", spinner="dots"):
                response = self.chat_engine.chat(user_input)
            
            console.print(Panel(Text(response, style="white"), title=f"[bold cyan]{self.personality['name']}[/bold cyan]", border_style="cyan"))
            console.print()

if __name__ == "__main__":
    tui = SuzoniTUI()
    tui.run()
