import os
import io
import json
from datetime import datetime
import re
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from atsuite_sdk.abstract import registry
from atsuite_sdk.events import emit_event
from atsuite_sdk.state import session_path
from claude_code_review_state import CLAUDE_CODE_REVIEW_STATE


def _ensure_repo() -> Path:
    return CLAUDE_CODE_REVIEW_STATE.repo_root_path()


def _run_git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_dir}", *args],
        cwd=repo_dir,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _require_pr(pr_id: int) -> dict[str, Any]:
    return CLAUDE_CODE_REVIEW_STATE.require_pr(pr_id)


def _normalize_diff(diff: str) -> str:
    lines = []
    for line in diff.splitlines():
        if line.startswith("index "):
            continue
        if line.startswith("@@"):
            marker = line.split("@@", 2)
            if len(marker) >= 2:
                line = f"@@{marker[1]}@@"
        lines.append(line)
    return "\n".join(lines).strip()


def _parse_commits(raw: str) -> list[dict[str, str]]:
    commits = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, message, authored_at, author = line.split("\x1f")
        commits.append(
            {
                "sha": sha,
                "message": message,
                "authored_at": datetime.fromisoformat(authored_at).isoformat(),
                "author": author,
            }
        )
    return commits


def _parse_name_status(raw: str) -> list[dict[str, str]]:
    status_map = {
        "A": "added",
        "M": "modified",
        "D": "deleted",
        "R": "renamed",
        "C": "copied",
    }
    files = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        files.append({"path": path, "status": status_map.get(status[0], status.lower())})
    return files


def _fixture_patch_for_pr(pr_payload: dict[str, Any], path: str = "") -> str:
    diffs = []
    for commit in pr_payload.get("commits", []):
        for patch in commit.get("patches", []):
            if path and str(patch["path"]) != path:
                continue
            diffs.append(str(patch["diff"]).rstrip())
    if not diffs:
        raise FileNotFoundError(path or f"patches for PR {pr_payload['id']}")
    return "\n".join(diffs).strip() + "\n"


def _git_log_range(repo_root: Path, base_ref: str, head_ref: str) -> list[dict[str, str]]:
    raw = _run_git(
        repo_root,
        "log",
        "--reverse",
        "--date=iso-strict",
        "--format=%H%x1f%s%x1f%aI%x1f%an",
        f"{base_ref}..{head_ref}",
    )
    return _parse_commits(raw)


def _resolve_commit_sha(repo_root: Path, abbreviated_sha: str) -> str:
    cleaned = abbreviated_sha.lstrip("^")
    return _run_git(repo_root, "rev-parse", cleaned).strip()


@registry.tool(access="r", domain="review")
def repo_git_list_pull_requests() -> dict[str, Any]:
    """List every pull request available in the deterministic review fixture."""
    repo_root = _ensure_repo()
    pull_requests = []
    for pr_id_str in sorted(CLAUDE_CODE_REVIEW_STATE.pull_requests, key=int):
        pr_id = int(pr_id_str)
        pr = _require_pr(pr_id)
        head_sha = _run_git(repo_root, "rev-parse", pr["head_ref"]).strip()
        pull_requests.append(
            {
                "id": pr_id,
                "title": str(pr["title"]),
                "topic": str(pr["topic"]),
                "base_ref": str(pr["base_ref"]),
                "head_ref": str(pr["head_ref"]),
                "head_sha": head_sha,
                "changed_files_count": len(pr.get("changed_files", [])),
            }
        )
    return {"pull_requests": pull_requests}


@registry.tool(access="r", domain="review")
def repo_git_get_pr_overview(pr_id: int) -> dict[str, Any]:
    """Return the metadata and actual commit list for a pull request."""
    repo_root = _ensure_repo()
    pr = _require_pr(pr_id)
    commits = _git_log_range(repo_root, pr["base_ref"], pr["head_ref"])
    return {
        "pr_id": pr_id,
        "title": str(pr["title"]),
        "topic": str(pr["topic"]),
        "summary": str(pr["summary"]),
        "base_ref": str(pr["base_ref"]),
        "head_ref": str(pr["head_ref"]),
        "review_focus": [str(item) for item in pr.get("review_focus", [])],
        "commits": commits,
        "changed_files": [dict(item) for item in pr.get("changed_files", [])],
    }


@registry.tool(access="r", domain="review")
def repo_git_list_changed_files(pr_id: int) -> dict[str, Any]:
    """List the changed files for a pull request using the materialized git refs."""
    repo_root = _ensure_repo()
    pr = _require_pr(pr_id)
    files = _parse_name_status(
        _run_git(repo_root, "diff", "--name-status", f"{pr['base_ref']}..{pr['head_ref']}")
    )
    return {
        "pr_id": pr_id,
        "base_ref": str(pr["base_ref"]),
        "head_ref": str(pr["head_ref"]),
        "files": files,
    }


@registry.tool(access="r", domain="review")
def repo_git_get_patch(pr_id: int, path: str = "") -> dict[str, Any]:
    """Return the structured patch stored for a pull request."""
    pr = _require_pr(pr_id)
    diff = _fixture_patch_for_pr(pr, path=path)
    files = [item["path"] for item in pr.get("changed_files", []) if not path or item["path"] == path]
    return {
        "pr_id": pr_id,
        "path": path,
        "base_ref": str(pr["base_ref"]),
        "head_ref": str(pr["head_ref"]),
        "files": files,
        "diff": diff,
        "normalized_diff": _normalize_diff(diff),
    }


@registry.tool(access="r", domain="review")
def repo_git_diff_between_refs(base_ref: str, head_ref: str, path: str = "") -> dict[str, Any]:
    """Return a git diff between two refs, optionally scoped to one path."""
    repo_root = _ensure_repo()
    command = ["diff", f"{base_ref}..{head_ref}"]
    if path:
        command.extend(["--", path])
    diff = _run_git(repo_root, *command)
    return {
        "base_ref": base_ref,
        "head_ref": head_ref,
        "path": path,
        "diff": diff,
        "normalized_diff": _normalize_diff(diff),
    }


@registry.tool(access="r", domain="review")
def repo_git_show_file_at_ref(path: str, ref: str) -> dict[str, Any]:
    """Read the contents of a file at a specific git ref."""
    repo_root = _ensure_repo()
    content = _run_git(repo_root, "show", f"{ref}:{path}")
    return {"path": path, "ref": ref, "content": content}


@registry.tool(access="r", domain="review")
def repo_git_log_for_path(path: str, ref: str = "main", limit: int = 5) -> dict[str, Any]:
    """Return commit history for a file path at a given ref."""
    repo_root = _ensure_repo()
    raw = _run_git(
        repo_root,
        "log",
        f"-n{limit}",
        "--date=iso-strict",
        "--format=%H%x1f%s%x1f%aI%x1f%an",
        ref,
        "--",
        path,
    )
    return {"path": path, "ref": ref, "commits": _parse_commits(raw)}


@registry.tool(access="r", domain="review")
def repo_git_blame_range(path: str, ref: str, start_line: int, end_line: int) -> dict[str, Any]:
    """Attribute a range of lines in a file to commits."""
    repo_root = _ensure_repo()
    raw = _run_git(
        repo_root,
        "blame",
        "-L",
        f"{start_line},{end_line}",
        ref,
        "--",
        path,
    )
    lines = []
    pattern = re.compile(r"^([0-9a-f^]+)\s+\((.+?)\s+\d{4}-\d{2}-\d{2}.*?\)\s(.*)$")
    current_line = start_line
    for line in raw.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        commit_sha, author, content = match.groups()
        full_sha = _resolve_commit_sha(repo_root, commit_sha)
        lines.append(
            {
                "line_number": current_line,
                "commit_sha": full_sha,
                "author": author.strip(),
                "content": content,
            }
        )
        current_line += 1
    return {
        "path": path,
        "ref": ref,
        "start_line": start_line,
        "end_line": end_line,
        "lines": lines,
    }


@registry.tool(access="r", domain="review")
def repo_git_show_commit(commit_sha: str) -> dict[str, Any]:
    """Show metadata and patch text for a commit."""
    repo_root = _ensure_repo()
    raw = _run_git(
        repo_root,
        "show",
        "--stat",
        "--format=%H%x1f%s%x1f%aI%x1f%an",
        "--patch",
        commit_sha,
    )
    first_line, _, diff = raw.partition("\n")
    sha, message, authored_at, author = first_line.split("\x1f")
    return {
        "commit": {
            "sha": sha,
            "message": message,
            "authored_at": datetime.fromisoformat(authored_at).isoformat(),
            "author": author,
        },
        "diff": diff,
    }


@registry.tool(access="rw", domain="review")
def repo_git_select_repository(project: str) -> dict:
    """Select a packaged project for this request's isolated review Session."""
    import json
    root = Path(os.environ['ATSUITE_REVIEW_PROJECTS']).resolve()
    projects = json.loads((root / 'projects.json').read_text())
    if project not in projects:
        raise ValueError('Unknown packaged review project')
    directory = (root / projects[project]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Project path escapes packaged fixtures')
    CLAUDE_CODE_REVIEW_STATE.set_candidate_data_dirs([directory])
    return {'project':project, 'pull_requests':sorted(CLAUDE_CODE_REVIEW_STATE.pull_requests)}


@registry.tool(access="rw", domain="review")
def repo_git_run_tests(pr_id: int, ref: str, suite: str, uid: str | None = None) -> dict[str, Any]:
    """Run one fixture-approved upstream regression suite on the PR base or head ref."""
    pr = _require_pr(pr_id)
    if ref not in {pr["base_ref"], pr["head_ref"]} or suite not in pr.get("test_suites", {}):
        raise ValueError("Choose a declared test suite and the PR base or head ref")
    repo = _ensure_repo()
    revision = _run_git(repo, "rev-parse", "--verify", ref + "^{commit}").strip()
    checkout = Path(session_path("test-checkouts/" + revision))
    if not checkout.exists():
        archive = subprocess.check_output(["git", "-c", f"safe.directory={repo}", "archive", revision], cwd=repo)
        checkout.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as incoming:
            incoming.extractall(checkout, filter="data")
    paths = pr["test_suites"][suite]
    if not isinstance(paths, list) or not paths:
        raise ValueError("The declared suite must contain test paths")
    for path in paths:
        if not isinstance(path, str) or path.startswith("-") or not (checkout / path).resolve().is_relative_to(checkout.resolve()):
            raise ValueError("Test path escapes the checkout")
    report = checkout / "e3-test-results.xml"
    report.unlink(missing_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(checkout), "LANG": "C.UTF-8",
        "PYTHONPATH": os.pathsep.join([str(checkout), *(p for p in sys.path if p)]),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider",
         "--junitxml=" + str(report), *paths],
        cwd=checkout, env=env, capture_output=True, text=True, timeout=180,
    )
    emit_event("review_test_execution", uid=uid, pr_id=pr_id, ref=ref, suite=suite,
               compute_s=time.perf_counter() - started, exit_code=completed.returncode)
    if completed.returncode not in {0, 1} or not report.is_file():
        raise RuntimeError("Upstream tests did not complete: " + completed.stderr[-1500:] + completed.stdout[-1500:])
    xml = ET.parse(report).getroot()
    cases = list(xml.iter("testcase"))
    failures = sorted(
        case.attrib.get("classname", "") + "::" + case.attrib["name"]
        for case in cases if case.find("failure") is not None or case.find("error") is not None
    )
    result = {"pr_id": pr_id, "ref": ref, "revision": revision, "suite": suite,
              "exit_code": completed.returncode, "tests": len(cases), "failed_tests": failures,
              "skipped": sum(case.find("skipped") is not None for case in cases)}
    if not cases or all(case.find("skipped") is not None for case in cases):
        raise RuntimeError("The suite executed no real tests")
    runs = dict(getattr(CLAUDE_CODE_REVIEW_STATE, "test_runs", {}))
    runs[f"{pr_id}:{ref}:{suite}"] = result
    CLAUDE_CODE_REVIEW_STATE.test_runs = runs
    return result


@registry.tool(access="r", domain="review")
def repo_git_get_test_results(pr_id: int) -> dict[str, Any]:
    """Read actual completed regression results retained in this request's Session."""
    _require_pr(pr_id)
    return {key: value for key, value in getattr(CLAUDE_CODE_REVIEW_STATE, "test_runs", {}).items()
            if key.startswith(str(pr_id) + ":")}
