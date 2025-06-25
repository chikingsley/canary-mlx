#!/usr/bin/env python3
"""
Setup script for Canary conversion environment.

This script sets up the environment for converting Canary models to MLX.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    try:
        console.print(f"[yellow]{description}...[/yellow]")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        console.print(f"[green]✅ {description} completed[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ {description} failed:[/red]")
        console.print(f"[red]{e.stderr}[/red]")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        console.print("[red]❌ Python 3.10+ is required[/red]")
        return False
    
    console.print(f"[green]✅ Python {version.major}.{version.minor}.{version.micro}[/green]")
    return True


def install_dependencies():
    """Install required dependencies."""
    console.print("[bold blue]Installing dependencies...[/bold blue]")
    
    # Install main requirements
    if not run_command(
        f"{sys.executable} -m pip install -r requirements-canary.txt",
        "Installing Python dependencies"
    ):
        return False
    
    # Install parakeet-mlx in development mode
    parakeet_path = Path("parakeet-mlx")
    if parakeet_path.exists():
        if not run_command(
            f"{sys.executable} -m pip install -e {parakeet_path}",
            "Installing parakeet-mlx in development mode"
        ):
            return False
    else:
        console.print("[yellow]⚠️  parakeet-mlx directory not found, skipping dev install[/yellow]")
    
    return True


def verify_installation():
    """Verify that all required packages are installed."""
    console.print("[bold blue]Verifying installation...[/bold blue]")
    
    required_packages = [
        "torch",
        "safetensors", 
        "pydantic",
        "typer",
        "rich",
        "mlx"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            console.print(f"[green]✅ {package}[/green]")
        except ImportError:
            console.print(f"[red]❌ {package}[/red]")
            failed_imports.append(package)
    
    if failed_imports:
        console.print(f"[red]Failed to import: {', '.join(failed_imports)}[/red]")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    console.print("[bold blue]Creating directories...[/bold blue]")
    
    directories = [
        "canary-mlx",
        "canary_nemo_extracted", 
        "logs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        console.print(f"[green]✅ Created {dir_name}/[/green]")


def download_model_info():
    """Display information about downloading the model."""
    console.print("\n[bold blue]Model Download Instructions:[/bold blue]")
    console.print("To download the Canary model, run:")
    console.print("[cyan]huggingface-cli download nvidia/canary-1b-flash --local-dir ./canary-1b-flash[/cyan]")
    console.print("\nOr use Python:")
    console.print("[cyan]from huggingface_hub import snapshot_download[/cyan]")
    console.print("[cyan]snapshot_download('nvidia/canary-1b-flash', local_dir='./canary-1b-flash')[/cyan]")


def main():
    """Run the setup process."""
    console.print("[bold green]Canary to MLX Conversion Setup[/bold green]\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        console.print("[red]❌ Dependency installation failed[/red]")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        console.print("[red]❌ Installation verification failed[/red]")
        sys.exit(1)
    
    # Show model download info
    download_model_info()
    
    console.print("\n[bold green]🎉 Setup completed successfully![/bold green]")
    console.print("\nNext steps:")
    console.print("1. Download the Canary model (see instructions above)")
    console.print("2. Run: [cyan]python test_canary_conversion.py[/cyan]")
    console.print("3. Run: [cyan]python canary_to_mlx_converter.py canary-1b-flash.nemo[/cyan]")


if __name__ == "__main__":
    main()