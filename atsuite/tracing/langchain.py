"""LangChain v1 create_agent uses the LangGraph execution and checkpoint adapter."""

from atsuite.tracing.langgraph import LangGraphTracingAdapter


class LangChainTracingAdapter(LangGraphTracingAdapter):
    """Capture create_agent's compiled graph without replacing its agent loop."""

    harness = "langchain"

    def _callback_free_dependencies(self, rid, previous):
        """create_agent hides router callbacks; use actual tool-call/result identities."""
        span = self.collector.spans[rid]
        inputs = span["input"]
        if not isinstance(inputs, dict):
            return None
        meta = self.tasks[rid]
        graph = self.graphs[self._graph_path(meta["graph_namespace"])]
        candidates = []
        for source in previous:
            branches = graph.builder.branches.get(self.tasks[source]["langgraph_node"], {})
            if any(
                not getattr(branch.path, "trace", True) and meta["langgraph_node"] in (branch.ends or {}).values()
                for branch in branches.values()
            ):
                candidates.append(source)
        if not candidates:
            return None

        if inputs.get("__type") == "tool_call_with_context":
            call_id = inputs["tool_call"]["id"]
            matches = []
            for source in candidates:
                prior = self.collector.spans[source]
                output = prior["output"]
                state = {
                    **(prior["input"] or {}),
                    **(output if isinstance(output, dict) else {}),
                }
                messages = state.get("messages", [])
                last_ai = next((m for m in reversed(messages) if m.get("type") == "ai"), {})
                if any(call.get("id") == call_id for call in last_ai.get("tool_calls", [])):
                    matches.append(source)
            if len(matches) == 1:
                span["metadata"]["routing_evidence"] = {"tool_call_id": call_id}
                return matches
        else:
            consumed = {m.get("tool_call_id") for m in inputs.get("messages", []) if isinstance(m, dict)} - {None}
            matches = []
            for source in candidates:
                output = self.collector.spans[source]["output"]
                messages = output.get("messages", []) if isinstance(output, dict) else []
                produced = {m.get("tool_call_id") for m in messages if isinstance(m, dict)} - {None}
                if produced and produced <= consumed:
                    matches.append(source)
            if matches:
                span["metadata"]["routing_evidence"] = {"consumed_tool_call_ids": sorted(consumed)}
                return matches
        return None
