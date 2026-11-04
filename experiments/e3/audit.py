"""Verify the built SDK, Tool sources, databases and Git fixtures before cloud profiling."""

import hashlib
import io
import json
import subprocess
import tarfile
import uuid
from pathlib import Path

from atsuite.pipeline import resolve_benchmark, target_image_tag


def audit(provider: str, config: Path) -> dict:
    resolved = resolve_benchmark(config, provider)
    target = next(iter(resolved.targets.values()))
    tag = target_image_tag(resolved.bench_name, target)
    metadata = json.loads(subprocess.check_output(["docker", "image", "inspect", tag], text=True))[0]
    container = "e3-audit-" + uuid.uuid4().hex[:12]
    subprocess.run(["docker", "create", "--name", container, tag], check=True, stdout=subprocess.DEVNULL)
    root = Path(__file__).resolve().parents[2]
    prefix = "/app/" if resolved.family == "faas" else "/app/mcp/Tools/"
    expected = {prefix + "implementation.py": resolved.bench_root / "nodes/Tools/implementation.py"}
    for name in ("state.py", "abstract.py", "function.py", "mcp_server.py"):
        expected["/app/atsuite_sdk/" + name] = root / "atsuite_sdk" / name
    payload = resolved.bench_root / "data/payload"
    # Check every packaged source and immutable database; include project Git object packs and refs.
    for source in payload.rglob("*"):
        if source.is_file():
            expected[prefix + "payload/" + source.relative_to(payload).as_posix()] = source
    hashes = {}
    try:
        # One archive read keeps a large fixture audit inexpensive.
        raw = subprocess.check_output(["docker", "cp", container + ":" + prefix + "payload", "-"])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    destination = prefix + member.name
                    if destination in expected:
                        hashes[destination] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        for destination, source in expected.items():
            if destination not in hashes:
                raw = subprocess.check_output(["docker", "cp", container + ":" + destination, "-"])
                with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                    member = next(m for m in archive.getmembers() if m.isfile())
                    hashes[destination] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if hashes[destination] != hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError("Image source/data mismatch: " + destination)
        subprocess.check_output(["docker", "cp", container + ":/usr/bin/git", "-"])
    finally:
        subprocess.run(["docker", "rm", container], check=True, stdout=subprocess.DEVNULL)
    return {
        "provider": provider,
        "tag": tag,
        "id": metadata["Id"],
        "architecture": metadata["Architecture"],
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "files": hashes,
    }
