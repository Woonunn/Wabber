"""Local mono-pool frontend for the unchanged TravelPlanner tools and Notebook."""

import importlib

for node in ("flights", "accommodations", "restaurants", "attractions", "cities", "google_distance_matrix", "notebook"):
    importlib.import_module(f"benchmarks.TravelPlanner.nodes.{node}.implementation")
