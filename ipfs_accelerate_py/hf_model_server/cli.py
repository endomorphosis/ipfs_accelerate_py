"""
Command-line interface for HF Model Server
"""

import click
import logging
from .server import create_server
from .config import ServerConfig


@click.group()
def cli():
    """Unified HuggingFace Model Server CLI"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option("--workers", default=1, type=int, help="Number of workers")
@click.option("--log-level", default="INFO", help="Log level")
@click.option("--config-file", type=click.Path(exists=True), help="Config file path")
def serve(host, port, workers, log_level, config_file):
    """Start the HF Model Server"""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create config
    if config_file:
        # TODO: Load from file
        config = ServerConfig()
    else:
        config = ServerConfig(
            host=host,
            port=port,
            workers=workers,
            log_level=log_level
        )
    
    # Create and run server
    server = create_server(config)
    click.echo(f"Starting HF Model Server on {host}:{port}")
    server.run()


@cli.command()
def discover():
    """Discover available HF skills"""
    from .registry import SkillRegistry
    import asyncio
    
    async def _discover():
        registry = SkillRegistry(
            skill_directories=["ipfs_accelerate_py"],
            skill_pattern="hf_*.py"
        )
        count = await registry.discover_skills()
        
        click.echo(f"\nDiscovered {count} skills:\n")
        for skill in registry.list_skills():
            click.echo(f"  - {skill.name}")
            click.echo(f"    Model: {skill.model_id}")
            click.echo(f"    Architecture: {skill.architecture}")
            click.echo(f"    Task: {skill.task_type}")
            click.echo(f"    Hardware: {', '.join(skill.supported_hardware)}")
            click.echo()
    
    asyncio.run(_discover())


@cli.command()
def hardware():
    """Show available hardware"""
    from .hardware import HardwareDetector
    
    detector = HardwareDetector()
    available = detector.get_available_hardware()
    
    click.echo(f"\nAvailable hardware: {', '.join(available)}\n")
    
    for hw_name in ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]:
        cap = detector.get_capability(hw_name)
        if cap and cap.available:
            click.echo(f"{hw_name.upper()}:")
            click.echo(f"  Devices: {cap.device_count}")
            if cap.memory_total_mb > 0:
                click.echo(f"  Memory: {cap.memory_total_mb:.0f} MB total, {cap.memory_available_mb:.0f} MB available")
            if cap.compute_capability:
                click.echo(f"  Compute: {cap.compute_capability}")
            click.echo()


if __name__ == "__main__":
    cli()
