"""Shared tool-using LangGraph mechanics; workload policy and validation stay separate."""

import asyncio
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from benchmarks.DataSciBench.langgraph_agent.tool_client import ToolExecutionError


class AgentState(TypedDict, total=False):
    user_prompt: str
    messages: list[dict]
    iteration: int
    status: str
    decision: dict
    answer: dict
    validation: dict


def build_graph(model, tools, task: dict, system: str, validate, max_iterations: int = 40):
    """Plan each next call from actual prior results; never prescribe a future DAG."""
    definitions = task["tool_definitions"]
    names = {d["name"] for d in definitions}
    if len(names) != len(definitions) or not names:
        raise ValueError("Unique tool definitions are required")
    instruction = (
        "Return only one valid JSON object. To call a tool, use keys tool (a tool name) and args (an object). "
        "To finish, use only the key answer (an object). Do not emit Markdown or placeholder values. "
        + system
        + '\nCall exactly one tool per response. Example: {"tool":"'
        + sorted(names)[0]
        + '","args":{}}.\nAvailable tool definitions: '
        + json.dumps(definitions, sort_keys=True)
    )

    async def decide(state):
        messages = state.get("messages") or [
            {"role": "system", "content": instruction},
            {"role": "user", "content": state["user_prompt"]},
        ]
        if state["iteration"] >= max_iterations:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "The tool-call budget is exhausted. Submit your final answer now; no further tools are available.",
                },
            ]
        decision = await model.decide("ACT", state["iteration"], messages)
        if set(decision) == {"answer"} and isinstance(decision["answer"], dict):
            return {"decision": decision, "messages": messages, "answer": decision["answer"], "status": "validating"}
        if state["iteration"] >= max_iterations:
            return {"status": "failed", "messages": messages}
        calls = decision.get("calls") if set(decision) == {"calls"} else [decision]
        if (
            not isinstance(calls, list)
            or not calls
            or any(
                not isinstance(call, dict)
                or set(call) != {"tool", "args"}
                or call["tool"] not in names
                or not isinstance(call["args"], dict)
                or any(k == "uid" or k.startswith("__atsuite_") for k in call["args"])
                for call in calls
            )
        ):
            return {
                "messages": [
                    *messages,
                    {"role": "assistant", "content": json.dumps(decision, sort_keys=True)},
                    {
                        "role": "user",
                        "content": "Invalid tool call. Use listed tool names and object args, with no uid.",
                    },
                ],
                "iteration": state["iteration"] + 1,
                "status": "invalid_action" if state["iteration"] + 1 < max_iterations else "failed",
            }
        return {"decision": decision, "messages": messages, "status": "running"}

    async def act(state):
        decision = state["decision"]
        if "calls" in decision:
            parallel = all(task.get("tool_access", {}).get(c["tool"]) in {"r", "stateless"} for c in decision["calls"])

            async def invoke(index, call):
                try:
                    return await tools.call(
                        call["tool"],
                        call["args"],
                        f"act-{state['iteration']}-{index}",
                        wave=f"batch-{state['iteration']}" if parallel else None,
                    )
                except ToolExecutionError as exc:
                    return {"tool_error": str(exc)}

            results = (
                await asyncio.gather(*(invoke(i, call) for i, call in enumerate(decision["calls"])))
                if parallel
                else [await invoke(i, call) for i, call in enumerate(decision["calls"])]
            )
            result = {"tool_results": results}
        else:
            try:
                result = {
                    "tool_result": await tools.call(decision["tool"], decision["args"], f"act-{state['iteration']}")
                }
            except ToolExecutionError as exc:
                result = {"tool_result": {"tool_error": str(exc)}}
        return {
            "messages": [
                *state["messages"],
                {"role": "assistant", "content": json.dumps(decision, sort_keys=True)},
                {"role": "user", "content": json.dumps(result, sort_keys=True)},
            ],
            "iteration": state["iteration"] + 1,
            "status": "running",
        }

    async def verify(state):
        # Independent checks are never supplied to the model as planning hints.
        validation = await validate(task, state["answer"], tools)
        return {"validation": validation, "status": "complete" if validation["ok"] else "failed"}

    graph = StateGraph(AgentState)
    graph.add_node("DECIDE", decide)
    graph.add_node("TOOL", act)
    graph.add_node("VERIFY", verify)
    graph.add_edge(START, "DECIDE")
    graph.add_conditional_edges(
        "DECIDE",
        lambda s: (
            END
            if s["status"] == "failed"
            else "DECIDE"
            if s["status"] == "invalid_action"
            else "VERIFY"
            if s["status"] == "validating"
            else "TOOL"
        ),
    )
    graph.add_conditional_edges("TOOL", lambda s: END if s["status"] == "failed" else "DECIDE")
    graph.add_edge("VERIFY", END)
    return graph.compile()


def result_view(value):
    """Compare complete business results, including strings, lists and numeric values."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value
