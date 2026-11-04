"""Review agent with independent defect-location scoring."""

from langgraph.graph import END, START, StateGraph

from experiments.e3.agent import AgentState
from experiments.e3.agent import build_graph as build_tool_graph
from experiments.e3.agent import result_view


async def validate(task, answer, tools):
    """Match known introduced defects, rejecting approval and out-of-diff findings."""
    expected = task["verification"]["defects"]
    findings = answer.get("findings", [])
    if not isinstance(findings, list):
        return {"ok": False, "basis": "known defects"}
    matched = [
        any(
            isinstance(f, dict)
            and f.get("file") == d["file"]
            and type(f.get("line")) is int
            and d["start"] <= f["line"] <= d["end"]
            and isinstance(f.get("body"), str)
            and bool(f["body"].strip())
            and (not d.get("concepts") or any(c in f["body"].lower() for c in d["concepts"]))
            for f in findings
        )
        for d in expected
    ]
    valid = all(
        isinstance(f, dict)
        and any(
            f.get("file") == d["file"] and type(f.get("line")) is int and d["start"] <= f["line"] <= d["end"]
            for d in expected
        )
        for f in findings
    )
    result = {
        "ok": all(matched) and valid and answer.get("verdict") == ("changes_requested" if expected else "approve"),
        "basis": "sealed introduced-defect locations (fixture tests are checked during preparation)",
        "matched": matched,
    }
    if task["verification"].get("required_test_suites"):
        evidence = result_view(await tools.call("repo_git_get_test_results", {"pr_id": task["pr_id"]}, "verify-test-evidence"))
        for suite in task["verification"]["required_test_suites"]:
            base = evidence.get(f"{task['pr_id']}:{task['base_ref']}:{suite}", {})
            head = evidence.get(f"{task['pr_id']}:{task['head_ref']}:{suite}", {})
            result["ok"] &= bool(
                base.get("exit_code") == 0 and base.get("tests", 0) > base.get("skipped", 0)
                and head.get("exit_code") == 1 and head.get("failed_tests")
                and base.get("revision") == task["verification"]["base_revision"]
                and head.get("revision") == task["verification"]["head_revision"]
            )
        result["basis"] += "; actual completed upstream tests pass at base and fail at head"
    return result


def build_graph(model, tools, task, max_iterations=40):
    agent = build_tool_graph(
        model,
        tools,
        task,
        "Review the requested PR. Inspect files and Git diff and verify findings against refs. "
        "Once you have concrete reproduction evidence, submit the findings; avoid inspecting unrelated files. "
        'Report {"verdict":"approve|changes_requested", "findings":[{"file":"...","line":1,"body":"..."}]} as answer.',
        validate,
        max_iterations,
    )

    if not task.get("project"):
        return agent

    async def select(state):
        await tools.call("repo_git_select_repository", {"project": task["project"]}, "select-project")
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("PROJECT", select)
    graph.add_node("REVIEW", agent)
    graph.add_edge(START, "PROJECT")
    graph.add_edge("PROJECT", "REVIEW")
    graph.add_edge("REVIEW", END)
    return graph.compile()
