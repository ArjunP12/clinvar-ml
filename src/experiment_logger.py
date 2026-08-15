"""Simple experiment logger for reproducibility."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

from typing import Any, Dict


def log_experiment(config: dict[str, Any], metrics: dict[str, Any], output_path: Path) -> None:
    """Save experiment configuration and results to JSON.
    
    Args:
        config: Dictionary of experiment configuration.
        metrics: Dictionary of result metrics.
        output_path: Path to save the experiment log.
    """
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=config.get("project_root", "."),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "config": _convert_keys(config),
        "metrics": _convert_keys(metrics),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)


def _convert_keys(obj):
    """Recursively convert dict keys to strings."""
    if isinstance(obj, dict):
        return {str(k): _convert_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_keys(item) for item in obj]
    return obj
