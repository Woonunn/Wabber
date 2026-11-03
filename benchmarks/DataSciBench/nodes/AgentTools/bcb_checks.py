"""Pinned upstream BCB calls and tests; execute only inside the Tool sandbox."""

import ast
import gzip
import hashlib
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from joblib.externals import cloudpickle


def load_case(task: str) -> dict:
    """Read the immutable evaluation bundle installed in the sandbox image."""
    path = Path(__file__).with_name("bcb_cases.json.gz")
    with gzip.open(path, "rt") as handle:
        return json.load(handle)["cases"][task]


def seed_inputs() -> None:
    """Freeze fixture randomness without changing the upstream fixture code."""
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    try:
        from faker import Faker
    except ImportError:
        return
    Faker.seed(0)


def prepare(case: dict, directory: Path) -> None:
    """Materialize the original upstream invocation and serialize its arguments."""
    seed_inputs()
    scope = {"__name__": "bcb_fixture"}
    # Upstream evaluation supplies library aliases before executing its input snippets.
    imports = [node for node in ast.parse(case["reference"]).body
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=imports, type_ignores=[]), "fixture_imports.py", "exec"), scope)
    exec(compile(case["data"], "upstream_fixture.py", "exec"), scope)
    args = scope["args"]
    if not isinstance(args, (tuple, list)):
        raise ValueError("Upstream args must be a sequence")
    check_input_paths(args)
    (directory / "fixture_args.pkl").write_bytes(cloudpickle.dumps(args, protocol=5))


def check_input_paths(value: object) -> None:
    """Reject transient absolute paths that cannot be restored from sealed inputs."""
    if isinstance(value, (str, Path)) and Path(value).is_absolute():
        raise ValueError("Upstream fixture contains an unsealed absolute path: " + str(value))
    if isinstance(value, (tuple, list)):
        for item in value:
            check_input_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            check_input_paths(item)


def invoke(source: str, inputs: Path, *, timing_observations: dict | None = None) -> object:
    """Actually call the generated function on the frozen upstream input."""
    seed_inputs()
    args = cloudpickle.loads((inputs / "fixture_args.pkl").read_bytes())
    check_input_paths(args)
    scope = {"__name__": "bcb_solution"}
    exec(compile(source, "solution.py", "exec"), scope)
    if timing_observations is None:
        return scope["task_func"](*args)
    import inspect
    import time

    bound = inspect.signature(scope["task_func"]).bind(*args)
    bound.apply_defaults()
    timing_observations.update(samples=bound.arguments["samples"], delay=bound.arguments["delay"],
                               timestamps=[], sleeps=[])
    clock, sleep = time.time, time.sleep

    def observed_clock():
        value = clock()
        timing_observations["timestamps"].append(value)
        return value

    def observed_sleep(delay):
        timing_observations["sleeps"].append(delay)
        return sleep(delay)

    try:
        time.time, time.sleep = observed_clock, observed_sleep
        return scope["task_func"](*args)
    finally:
        time.time, time.sleep = clock, sleep


def timing_signature(value: object, evidence: dict) -> str:
    """Recompute bcb823's actual timing statistics; never substitute a fixed duration."""
    import math
    import numpy as np
    from sandbox_worker import digest

    count, delay = evidence["samples"], evidence["delay"]
    stamps = evidence["timestamps"]
    if (type(count) is not int or count <= 0 or not math.isfinite(delay) or delay < 0
            or len(stamps) != count * 2 or evidence["sleeps"] != [delay] * count
            or not all(math.isfinite(t) for t in stamps)):
        raise ValueError("Timing invocation evidence is incomplete")
    elapsed = np.array(stamps[1::2]) - np.array(stamps[::2])
    if np.any(elapsed < delay):
        raise ValueError("Observed sleep was shorter than requested")
    expected = (float(np.mean(elapsed)), float(np.std(elapsed)))
    if type(value) is not tuple or len(value) != 2 or value != expected:
        raise ValueError("Returned timing statistics differ from independent observations")
    return digest({"comparison": "observed-sleep-statistics-v1", "samples": count, "delay": delay})


def run_tests(source: str, case: dict) -> dict:
    """Require the entire original test suite to pass, with no skipped tests."""
    import matplotlib.pyplot as plt

    # Deserializing a returned Figure for its signature must not seed independent test state.
    plt.close("all")
    seed_inputs()
    scope = {"__name__": "bcb_solution"}
    exec(compile(source, "solution.py", "exec"), scope)
    function = scope["task_func"]
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return function(*args, **kwargs)

    scope["task_func"] = counted
    exec(compile(case["testcase"], "upstream_tests.py", "exec"), scope)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(scope["TestCases"])
    expected = suite.countTestCases()
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if (not result.wasSuccessful() or result.skipped or result.expectedFailures
            or result.testsRun != expected or expected == 0 or calls == 0):
        raise ValueError("Independent BCB tests failed:\n" + log.getvalue()[-12000:])
    return {"tests_run": result.testsRun, "function_calls": calls,
            "test_sha256": hashlib.sha256(case["testcase"].encode()).hexdigest(),
            "solution_sha256": hashlib.sha256(source.encode()).hexdigest()}


def return_signature(value: object, *, clock_window: list[float] | None = None) -> str:
    """Compare numerical returns and full rendered figures, excluding pickle object identities."""
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sandbox_worker import canonical, digest

    def view(item):
        figure = item if isinstance(item, Figure) else item.figure if isinstance(item, Artist) else None
        if type(item).__module__.startswith("seaborn."):
            figure = item.figure
        if isinstance(figure, Figure):
            annotations = []
            if clock_window is not None:
                from datetime import datetime
                import math
                from matplotlib.text import Annotation

                start, end = clock_window
                if not all(math.isfinite(v) for v in (start, end)) or end < start:
                    raise ValueError("Invalid invocation clock evidence")
                annotations = [(a, a.get_text()) for a in figure.findobj(Annotation)]
                if len(annotations) != 1:
                    raise ValueError("Expected exactly one current-time annotation")
                text = annotations[0][1]
                stamp = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
                if (stamp.strftime("%Y-%m-%d %H:%M:%S") != text
                        or not math.floor(start) <= stamp.timestamp() <= end):
                    raise ValueError("Annotation does not match the actual invocation time")
            # Only the validated clock text varies; original pickle bytes are retained.
            try:
                for annotation, _ in annotations:
                    annotation.set_text("2000-01-01 00:00:00")
                canvas = FigureCanvasAgg(figure)
                canvas.draw()
                pixels = hashlib.sha256(canvas.buffer_rgba()).hexdigest()
            finally:
                for annotation, text in annotations:
                    annotation.set_text(text)
            geometry = {name: canonical(getattr(item, name)()) for name in (
                "get_array", "get_offsets", "get_xdata", "get_ydata", "get_xy", "get_height", "get_width"
            ) if callable(getattr(item, name, None))}
            return {"type": type(item).__module__ + "." + type(item).__name__,
                    **({"geometry": geometry} if geometry else {}),
                    "size": list(canvas.get_width_height()),
                    "rgba_sha256": pixels}
        if isinstance(item, (tuple, list)):
            return {"type": type(item).__name__, "items": [view(v) for v in item]}
        if isinstance(item, dict):
            return {str(k): view(v) for k, v in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if type(item).__module__.startswith("statsmodels.") and hasattr(item, "params"):
            return {"type": type(item).__name__, "params": canonical(item.params),
                    "fittedvalues": canonical(item.fittedvalues), "resid": canonical(item.resid)}
        return canonical(item)

    return digest(view(value))


def validate(task: str, artifacts: Path, inputs: Path) -> dict:
    """Validate the executed solution in a fresh worker, using upstream tests."""
    source = (artifacts / "solution.py").read_text()
    receipt = json.loads((artifacts / "bcb_execution.json").read_text())
    if receipt != {"solution_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "input_sha256": hashlib.sha256((inputs / "fixture_args.pkl").read_bytes()).hexdigest(),
                   "return_sha256": hashlib.sha256((artifacts / "bcb_result.pkl").read_bytes()).hexdigest(),
                   "invocations": 1}:
        raise ValueError("BCB execution receipt does not match solution and inputs")
    original = Path.cwd()
    # Upstream test fixtures may delete their directories; never alias prepared inputs there.
    with tempfile.TemporaryDirectory(prefix="independent-tests-", dir=original) as temporary:
        try:
            os.chdir(temporary)
            checks = run_tests(source, load_case(task))
        finally:
            os.chdir(original)
    checks["original_input_invocations"] = 1
    return checks


def preflight(task: str, bundle: Path, output: Path) -> None:
    """Check the upstream reference and prepare inputs inside an isolated container."""
    import os
    import shutil
    import time
    import traceback

    with gzip.open(bundle, "rt") as handle:
        case = json.load(handle)["cases"][task]
    started = time.perf_counter()
    result = {"task_id": task, "status": "failed", "agent_execution": False}
    try:
        inputs = output / "inputs"
        inputs.mkdir()
        os.chdir(inputs)
        prepare(case, inputs)
        work = output / "reference"
        shutil.copytree(inputs, work)
        os.chdir(work)
        invoke(case["reference"], inputs)
        tests = output / "tests"
        tests.mkdir()
        os.chdir(tests)
        checks = run_tests(case["reference"], case)
        result.update(status="reference_validated", checks=checks)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result["wall_s"] = time.perf_counter() - started
    (output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    import sys

    preflight(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
