"""Pinned real projects with explicit injected PRs and offline regression oracles."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECTS = {
    "requests": {
        "url": "https://github.com/psf/requests.git",
        "revision": "0e322af87745eff34caffe4df68456ebc20d9068",
        "file": "src/requests/sessions.py",
        "old": "if v is None]",
        "new": "if v is not None]",
        "test": "from requests.sessions import merge_setting\nassert merge_setting({'X':None}, {'X':'old','Y':'keep'}) == {'Y':'keep'}\n",
        "concepts": ["none", "null", "remove", "delet", "header", "setting"],
        "summary": "Refactor request and session setting merge",
    },
    "itsdangerous": {
        "url": "https://github.com/pallets/itsdangerous.git",
        "revision": "096c8d42545d3b68ea21a4f890fb2b2d8979c0bd",
        "file": "src/itsdangerous/timed.py",
        "old": "if age > max_age:",
        "new": "if age < max_age:",
        "test": "from itsdangerous import TimestampSigner, SignatureExpired\nclass Clock(TimestampSigner):\n    now=1000\n    def get_timestamp(self): return self.now\ns=Clock('fixture-key'); token=s.sign(b'payload'); s.now=1011\ntry: s.unsign(token,max_age=10)\nexcept SignatureExpired: pass\nelse: raise AssertionError('expired signature accepted')\n",
        "concepts": ["expir", "age", "older", "timestamp", "max_age", "valid"],
        "summary": "Simplify signed-token age validation",
    },
    "click": {
        "url": "https://github.com/pallets/click.git",
        "revision": "934813e4d421071a1b3db3973c02fe2721359a6e",
        "file": "src/click/formatting.py",
        "old": "term_len(col)",
        "new": "len(col)",
        "test": "from click.formatting import measure_table\nassert measure_table([('\\x1b[31mred\\x1b[0m','x')]) == (3,1)\n",
        "concepts": ["ansi", "escap", "color", "colour", "terminal", "width"],
        "summary": "Streamline help-table column measurement",
    },
}


SECONDARY = {
    "requests": {
        "file": "src/requests/utils.py",
        "old": "return max(0, total_length - current_position)",
        "new": "return max(0, total_length)",
        "test": "import io\nfrom requests.utils import super_len\ns=io.BytesIO(b'abcdef');s.seek(2)\nassert super_len(s)==4\n",
        "concepts": ["position", "remain", "length", "seek", "offset", "content-length"],
        "summary": "Simplify stream length bookkeeping",
    },
    "itsdangerous": {
        "file": "src/itsdangerous/url_safe.py",
        "old": "if decompress:",
        "new": "if not decompress:",
        "test": "from itsdangerous import URLSafeSerializer\ns=URLSafeSerializer('fixture-key');data={'message':'abc'*1000};token=s.dumps(data)\ntry: actual=s.loads(token)\nexcept Exception: actual=None\nassert actual==data, 'URL-safe serializer round trip failed'\n",
        "concepts": ["compress", "payload", "round", "decode", "zlib"],
        "summary": "Refactor URL-safe payload decoding",
    },
    "click": {
        "file": "src/click/types.py",
        "old": "operator.ge if self.max_open else operator.gt",
        "new": "operator.gt if self.max_open else operator.ge",
        "test": "from click import IntRange\ntry: actual=IntRange(0,10).convert('10',None,None)\nexcept Exception: actual=None\nassert actual==10, 'inclusive upper bound rejected'\n",
        "concepts": ["bound", "inclusive", "range", "max_open", "operator", "equal"],
        "summary": "Normalize numeric upper-bound comparisons",
    },
}

COMPUTE_PROJECTS = {
    "sympy-polys": {
        "url": "https://github.com/sympy/sympy.git",
        "revision": "c870d17c2638f5061800e344b72ff80086a0b41d",
        "file": "sympy/polys/polytools.py",
        "old": "return _keep_coeff(coeff, _factors_product(factors))",
        "new": "return _factors_product(factors)",
        "test": "from sympy import factor, symbols\nx=symbols('x'); assert factor(2*x+2).expand() == 2*x+2\n",
        "concepts": ["coeff", "constant", "factor", "scalar", "multipli"],
        "summary": "Simplify symbolic factor reconstruction",
        "suites": {"polynomial-regression": ["sympy/polys/tests/test_polytools.py"]},
    },
    "sympy-matrices": {
        "url": "https://github.com/sympy/sympy.git",
        "revision": "c870d17c2638f5061800e344b72ff80086a0b41d",
        "file": "sympy/matrices/eigen.py",
        "old": "all_eigs[val] = all_eigs.get(val, 0) + 1",
        "new": "all_eigs[val] = 1",
        "test": "from sympy import eye\nassert eye(3).eigenvals() == {1:3}\n",
        "concepts": ["multiplicity", "repeated", "duplicate", "count", "eigen"],
        "summary": "Simplify eigenvalue block aggregation",
        "suites": {"matrix-regression": ["sympy/matrices/tests/test_eigen.py", "sympy/matrices/tests/test_matrixbase.py"]},
    },
}

COMPUTE_SECONDARY = {
    "sympy-polys": {
        "file": "sympy/polys/polytools.py",
        "old": "return Mul(*[f.as_expr()**k for f, k in factors])",
        "new": "return Mul(*[f.as_expr() for f, k in factors])",
        "test": "from sympy import factor, symbols\nx=symbols('x'); assert factor(x*x+2*x+1).expand() == x*x+2*x+1\n",
        "concepts": ["exponent", "power", "multiplicity", "repeated", "factor"],
        "summary": "Streamline factored product assembly",
    },
    "sympy-matrices": {
        "file": "sympy/matrices/eigen.py",
        "old": "all_eigs.append(val)",
        "new": "all_eigs.append(-val)",
        "test": "from sympy import diag\nassert diag(2,3).eigenvals(multiple=True) == [2,3]\n",
        "concepts": ["negative", "sign", "negat", "eigen", "diagonal"],
        "summary": "Normalize scalar eigenvalue extraction",
    },
}


def prepare(sources: Path, output: Path, expanded: bool = False, compute: bool = False) -> dict:
    """Never mislabel injected changes as historical upstream PRs."""
    output.mkdir(parents=True, exist_ok=False)
    receipts = {}
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "E3 Fixture",
        "GIT_AUTHOR_EMAIL": "e3@example.invalid",
        "GIT_COMMITTER_NAME": "E3 Fixture",
        "GIT_COMMITTER_EMAIL": "e3@example.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    tasks = []
    projects = {**PROJECTS, **COMPUTE_PROJECTS} if compute else PROJECTS
    secondary = {**SECONDARY, **COMPUTE_SECONDARY} if compute else SECONDARY
    for index, (name, case) in enumerate(projects.items(), 201):
        source = sources / f"e3-{name}-upstream"
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != case["revision"] or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
            raise ValueError("Upstream checkout changed: " + name)
        folder = output / name
        folder.mkdir()
        repo = folder / "fixture_repo"
        subprocess.run(["git", "clone", "--no-hardlinks", "--quiet", str(source.resolve()), str(repo)], check=True)

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True)

        git("checkout", "-B", "main", case["revision"])
        test_env = {**env, "PYTHONPATH": str(repo / "src" if (repo / "src").exists() else repo), "PYTHONDONTWRITEBYTECODE": "1"}
        ordinal = index - 201
        variants = [(index, [case])]
        if expanded:
            variants += [(301 + 2 * ordinal, [secondary[name]]), (302 + 2 * ordinal, [case, secondary[name]])]
        pull_requests = []
        for pr_id, changes in variants:
            git("checkout", "main")
            for change in changes:
                base = subprocess.run(
                    [sys.executable, "-c", change["test"]], env=test_env, capture_output=True, text=True
                )
                if base.returncode:
                    raise ValueError("Base regression failed: " + base.stderr)
            git("checkout", "-b", f"pr/{pr_id}")
            defects, commits, failures = [], [], []
            for change in changes:
                path = repo / change["file"]
                text = path.read_text()
                if text.count(change["old"]) != 1:
                    raise ValueError("Mutation no longer uniquely matches upstream")
                line = text[: text.index(change["old"])].count("\n") + 1
                path.write_text(text.replace(change["old"], change["new"]))
                git("add", change["file"])
                git("commit", "-qm", change["summary"])
                sha = git("rev-parse", "HEAD").strip()
                commits.append(
                    {"sha": sha, "patches": [{"path": change["file"], "diff": git("show", "--format=", sha)}]}
                )
                defects.append({"file": change["file"], "start": line, "end": line, "concepts": change["concepts"]})
            for change in changes:
                head = subprocess.run(
                    [sys.executable, "-c", change["test"]], env=test_env, capture_output=True, text=True
                )
                if head.returncode == 0 or "AssertionError" not in head.stderr:
                    raise ValueError("Head must fail the intended independent assertion")
                failures.append(
                    {
                        "file": change["file"],
                        "stderr": head.stderr,
                        "test_sha256": hashlib.sha256(change["test"].encode()).hexdigest(),
                    }
                )
            head_sha = git("rev-parse", "HEAD").strip()
            diff = git("diff", "main", f"pr/{pr_id}")
            receipt = {
                "project": name,
                "upstream": case["url"],
                "revision": actual,
                "head": head_sha,
                "fixture_kind": "injected regression on real upstream project",
                "base_passed": True,
                "head_failed": True,
                "regressions": failures,
                "changed_files": len(changes),
                "commits": len(commits),
                "diff_lines": len(diff.splitlines()),
            }
            receipts[f"{name}-{pr_id}"] = receipt
            summary = "; ".join(c["summary"] for c in changes)
            pull_requests.append(
                {
                    "id": pr_id,
                    "title": summary,
                    "topic": name,
                    "summary": summary,
                    "base_ref": "main",
                    "head_ref": f"pr/{pr_id}",
                    "changed_files": [{"path": c["file"], "status": "modified"} for c in changes],
                    "commits": commits,
                    **({"test_suites": case["suites"]} if case.get("suites") else {}),
                }
            )
            prompt = f"Review PR {pr_id} in the {name} project. Inspect the patch, relevant files and Git refs. Explain any introduced correctness defects with concrete reproduction evidence."
            if case.get("suites"):
                prompt += (
                    " Run the declared upstream regression suites " + ", ".join(case["suites"])
                    + f" at both main and pr/{pr_id}. Compare actual failures with passing base results before "
                    "submitting the review; do not invent test outcomes."
                )
            tasks.append(
                {
                    "task_id": f"{name}-{pr_id}",
                    "project": name,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "readiness": {"inputs": {"files": []}},
                    "verification": {"defects": defects, "regression": receipt},
                    **({"pr_id": pr_id, "base_ref": "main", "head_ref": f"pr/{pr_id}"} if case.get("suites") else {}),
                }
            )
            if case.get("suites"):
                tasks[-1]["verification"].update(required_test_suites=list(case["suites"]),
                    base_revision=actual, head_revision=head_sha)
        git("checkout", "main")
        (folder / "repo_fixture.json").write_text(
            json.dumps({"source": {"upstream": case["url"], "revision": actual}}, indent=2) + "\n"
        )
        (folder / "pr_fixture.json").write_text(json.dumps({"pull_requests": pull_requests}, indent=2) + "\n")
    (output / "projects.json").write_text(json.dumps({k: k for k in projects}, indent=2) + "\n")
    (output / "regressions.json").write_text(json.dumps(receipts, indent=2) + "\n")
    # Only immutable fixture material is served; private regression expectations stay outside each repository.
    os.environ["ATSUITE_DATA_PATH"] = str((output / "requests").resolve())
    nodes = Path(__file__).resolve().parents[2] / "benchmarks/ClaudeCodeReview/nodes"
    sys.path.insert(0, str(nodes / "repo_git"))
    from experiments.e3.prepare import definitions

    defs = definitions([f"benchmarks.ClaudeCodeReview.nodes.{n}.implementation" for n in ("repo_git", "repo_files")])
    defs = [d for d in defs if d["name"] != "repo_git_select_repository"]
    from atsuite_sdk.abstract import registry
    access = {d["name"]: registry.functions[d["name"]].access for d in defs}
    for task in tasks:
        task["tool_definitions"] = defs
        task["tool_access"] = access
    manifest = {"tasks": tasks, "assets": [str(p.resolve()) for p in output.rglob("*") if p.is_file()]}
    (output / "ClaudeCodeReview.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--compute", action="store_true", help="Include pinned SymPy regression-test review tasks")
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output, args.expanded, args.compute)))


if __name__ == "__main__":
    main()
