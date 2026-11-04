"""Build request context from the running harness's pre-execution inputs."""

from atsuite.workload.descriptor import describe

DATASCI_TOOLS = [
    "prepare_workspace",
    "execute_analysis",
    "merge_artifacts",
    "validate_result",
]


def describe_request(
    task: dict, config: dict, workload: str, *, planning_mode="live"
) -> dict:
    """Require deployment-pinned identities; endpoints and tool names are not revisions."""
    definitions = task.get("tool_definitions", [])
    names = (
        DATASCI_TOOLS
        if workload == "DataSciBench"
        else [d["name"] for d in definitions]
    )
    identity = config.get("prediction_identity", {})
    tools = identity.get("tools", names)
    if sorted(t["name"] if isinstance(t, dict) else t for t in tools) != sorted(names):
        raise ValueError(
            "Prediction identity must describe the complete workload tool set"
        )
    if definitions and "tools" in identity:
        actual = {d["name"]: d for d in definitions}
        if any(t["definition"] != actual[t["name"]] for t in tools):
            raise ValueError(
                "Prediction tool definitions differ from the harness definitions"
            )
    return describe(
        prompt=task["prompt"],
        files=task["readiness"]["inputs"]["files"],
        tools=tools,
        harness=identity.get("harness"),
        model={"name": "deepseek-flash", **config.get("model", {})},
        control={
            "planning_mode": planning_mode,
            **{k: config[k] for k in ("max_iterations", "max_parallel") if k in config},
            "tool_timeout_s": config.get("tool", {}).get("timeout", 240),
            **{
                k: task[k]
                for k in ("planner_instruction", "validation_contract", "output_files")
                if k in task
            },
        },
        session={"messages": [], "objects": []},
    )
