"""One execution pool for the existing file and Git review tools."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/ClaudeCodeReview/nodes/repo_git"))
for node in ("repo_git", "repo_files"):
    importlib.import_module(f"benchmarks.ClaudeCodeReview.nodes.{node}.implementation")
