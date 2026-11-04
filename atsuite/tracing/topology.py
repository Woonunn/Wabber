"""Project causal execution traces onto tools, preserving parallel branches and joins."""

from collections import defaultdict


def tool_topology(trace: dict) -> dict:
    """Return tool instances and minimal causal edges; never infer edges from timestamps."""
    spans = {span["id"]: span for span in trace["spans"]}
    children = defaultdict(list)
    for span in spans.values():
        children[span["parent_id"]].append(span["id"])
    predecessors = {}
    visiting = set()

    def before(rid):
        if rid not in spans:
            return set()
        if rid in predecessors:
            return predecessors[rid]
        if rid in visiting:
            raise ValueError("Cycle in observed causal dependencies")
        visiting.add(rid)
        span = spans[rid]
        result = set(before(span["parent_id"]))
        for dep in span["dependencies"]:
            result.update(completed(dep))
        visiting.remove(rid)
        predecessors[rid] = result
        return result

    def completed(rid):
        span = spans[rid]
        if span["kind"] == "tool":
            return {rid}
        result = set(before(rid))
        for child in children[rid]:
            result.update(completed(child))
        return result

    tools = [span for span in spans.values() if span["kind"] == "tool"]
    edges = []
    for span in tools:
        deps = before(span["id"])
        redundant = set().union(*(before(dep) for dep in deps)) if deps else set()
        edges.extend({"source": dep, "target": span["id"]} for dep in sorted(deps - redundant))
    return {
        "request_id": trace["request_id"],
        "complete": trace["summary"]["complete"],
        "nodes": [
            {
                key: span[key]
                for key in (
                    "id",
                    "name",
                    "status",
                    "duration_s",
                    "resources",
                    "input",
                    "error",
                )
            }
            for span in tools
        ],
        "edges": edges,
    }
