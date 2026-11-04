"""Shared deployment packaging; the three workloads keep their own existing tools."""

import importlib
import os
import sys
from pathlib import Path

payload = Path(__file__).resolve().parent / "payload"
sys.path.insert(0, str(payload))
sys.path.insert(0, str(payload / "benchmarks/ClaudeCodeReview/nodes/repo_git"))
os.environ["ATSUITE_DATA_PATH"] = str(payload / "review/requests")
os.environ["ATSUITE_REVIEW_PROJECTS"] = str(payload / "review")
os.environ["ATSUITE_SCIENTIFIC_PROBLEMS"] = str(payload / "scientific")
os.environ["ATSUITE_ALLOW_STATE_BOOTSTRAP"] = "false"
os.chdir(payload / "travel")
for workload, nodes in {
    "TravelPlanner": (
        "flights",
        "accommodations",
        "restaurants",
        "attractions",
        "cities",
        "google_distance_matrix",
        "notebook",
    ),
    "ScientificComputation": ("tensor_workspace",),
    "ClaudeCodeReview": ("repo_git", "repo_files"),
}.items():
    for node in nodes:
        importlib.import_module(f"benchmarks.{workload}.nodes.{node}.implementation")
