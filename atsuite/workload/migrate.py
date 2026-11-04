"""Migrate audited raw arrival contexts; label-only rows stay quarantined."""

import argparse
import copy
import json
from pathlib import Path

from atsuite.tracing.storage import read_text, write_json
from atsuite.workload.descriptor import compatibility_key, validate_descriptor
from atsuite.workload.profile import number


def migrate(snapshot: dict, arrivals: dict) -> dict:
    """Attach reconstructed contexts only with pre-arrival provenance; retain every old row."""
    if snapshot.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported workload model schema")
    result = copy.deepcopy(snapshot)
    migrated = quarantined = 0
    for sample in result["samples"]:
        profile = sample["profile"]
        source = arrivals.get(profile["request_id"])
        if source is not None:
            context = source["context"]
            validate_descriptor(context)
            if (
                not compatibility_key(context)
                or not source.get("evidence")
                or not number(source.get("visible_ns"))
                or not number(profile.get("arrival_ns"))
                or source["visible_ns"] > profile["arrival_ns"]
            ):
                raise ValueError(
                    "Migration requires raw context, pinned identities and pre-arrival provenance"
                )
            profile["descriptor"] = copy.deepcopy(context)
            profile["context_provenance"] = {
                k: source[k] for k in ("visible_ns", "evidence")
            }
            migrated += 1
        elif not compatibility_key(profile.get("descriptor")):
            quarantined += 1
    result.update(
        schema_version=2,
        errors=[],
        migration={
            "migrated": migrated,
            "quarantined": quarantined,
            "policy": "Old labels are retained for audit, excluded from prediction",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--arrivals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new snapshot")
    write_json(
        args.output,
        migrate(
            json.loads(read_text(args.snapshot)), json.loads(read_text(args.arrivals))
        ),
    )


if __name__ == "__main__":
    main()
