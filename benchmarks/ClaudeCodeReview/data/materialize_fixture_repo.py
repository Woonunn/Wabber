"""Rebuild deterministic Git history and independently check the introduced defects."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CASES = (
    (
        "pagination",
        "def page(items, offset, limit):\n    return items[offset:offset + limit]\n",
        "def page(items, offset, limit):\n    return items[offset:limit]\n",
        "assert page(list(range(10)), 3, 2) == [3, 4]",
    ),
    (
        "cache",
        "def lookup(cache, key, loader):\n    if key in cache:\n        return cache[key]\n    return loader(key)\n",
        "def lookup(cache, key, loader):\n    if cache.get(key):\n        return cache[key]\n    return loader(key)\n",
        'assert lookup({"x": 0}, "x", lambda key: 99) == 0',
    ),
    (
        "authorization",
        "def can_read(owner, user, public):\n    return public or owner == user\n",
        "def can_read(owner, user, public):\n    return public or owner != user\n",
        'assert not can_read("alice", "bob", False)',
    ),
)


def materialize(output: Path) -> dict:
    """Generate one repository with three independent PRs; never overwrite a checkout."""
    output.mkdir(parents=True, exist_ok=True)
    repo = output / "fixture_repo"
    repo.mkdir(exist_ok=False)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "E3 Fixture",
        "GIT_AUTHOR_EMAIL": "e3@example.invalid",
        "GIT_COMMITTER_NAME": "E3 Fixture",
        "GIT_COMMITTER_EMAIL": "e3@example.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout

    git("init", "-b", "main")
    for name, base, head, check in CASES:
        (repo / f"{name}.py").write_text(base)
    (repo / "README.md").write_text("Application utilities: pagination, cache lookup, and private-resource access.\n")
    git("add", ".")
    git("commit", "-m", "Initial application utilities")
    base_sha = git("rev-parse", "HEAD").strip()
    prs, checks = [], []
    for index, (name, base, head, check) in enumerate(CASES, 101):
        for label, source, expected in [("base", base, 0), ("head", head, 1)]:
            result = subprocess.run([sys.executable, "-c", source + "\n" + check], capture_output=True)
            if (result.returncode != 0) != bool(expected):
                raise ValueError(f"Independent regression check failed: {name}/{label}")
        git("checkout", "-b", f"pr/{index}", "main")
        (repo / f"{name}.py").write_text(head)
        git("add", ".")
        git("commit", "-m", f"Update {name} behavior")
        sha = git("rev-parse", "HEAD").strip()
        git("update-ref", f"refs/pull/{index}/head", sha)
        diff = git("diff", "main", f"pr/{index}", "--", f"{name}.py")
        prs.append(
            {
                "id": index,
                "title": f"Update {name}",
                "topic": name,
                "summary": f"Review the {name} behavior change.",
                "base_ref": "main",
                "head_ref": f"pr/{index}",
                "changed_files": [{"path": f"{name}.py", "status": "modified"}],
                "commits": [{"sha": sha, "patches": [{"path": f"{name}.py", "diff": diff}]}],
            }
        )
        checks.append(
            {
                "pr_id": index,
                "file": f"{name}.py",
                "start": 2,
                "end": 2,
                "base_passed": True,
                "head_failed": True,
                "test": check,
            }
        )
    git("checkout", "main")
    payload = {"base_sha": base_sha, "checks": checks}
    (output / "repo_fixture.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "pr_fixture.json").write_text(json.dumps({"pull_requests": prs}, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir)))


if __name__ == "__main__":
    main()
