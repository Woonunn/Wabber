"""Travel agent; validation delegates to the pinned official offline evaluator."""

import asyncio
import json
import os

from experiments.e3.agent import build_graph as build_tool_graph


async def validate(task, answer, tools):
    """Use a prepared evaluator command, failing closed on missing database or evaluator."""
    command = task["verification"]["command"]
    if not isinstance(command, list) or not command or any(not isinstance(s, str) for s in command):
        raise ValueError("Prepared official evaluation command required")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1"},
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(json.dumps(answer).encode()), 120)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError("Official TravelPlanner evaluation failed: " + stderr.decode()[-2000:])
    result = json.loads(stdout)
    if type(result.get("ok")) is not bool or not result.get("basis"):
        raise ValueError("Invalid official evaluation receipt")
    return result


def build_graph(model, tools, task, max_iterations=40):
    return build_tool_graph(
        model,
        tools,
        task,
        task.get(
            "planner_instruction",
            "Plan the original trip using the offline query tools and Notebook. Use only returned database entries. ",
        )
        + "Collect the necessary data using tools, then finish with an answer object containing plan. "
        "Compare direct transport in both directions and affordable ground alternatives early. "
        "When a direct flight is unavailable, query ground transport before searching connecting flights; "
        "respect any no-driving constraint and the time available. Do not assume the trip requires flying. "
        "A complete trip returns to its origin on the final day. Before submitting, check every day, "
        "route continuity, unique restaurants and attractions, lodging minimum nights and capacity, "
        "house rules, room type, required cuisines and transportation restrictions. "
        "Compute the full party's round-trip transport, every meal and all lodging nights; "
        "multiply per-person prices by party size and reserve enough rooms for everyone. "
        "Choose affordable valid entries rather than leaving required activities empty. "
        "Use the Notebook to retain query information or a budget breakdown when useful. "
        "Use the official itinerary serialization: current_city is a city or from City to City; "
        "named meals, attractions and accommodations are Name, City; separate attractions with semicolons; "
        "flights are Flight Number: F123, from City to City, Departure Time: HH:MM, Arrival Time: HH:MM; "
        "for road transport query mode self-driving or taxi and retain that exact transport label. "
        "use - for absent entries. "
        'Return the itinerary as {"plan":[{"days":1,"current_city":"...",'
        '"transportation":"...","breakfast":"...","attraction":"...",'
        '"lunch":"...","dinner":"...","accommodation":"..."}]}.',
        validate,
        max_iterations,
    )
