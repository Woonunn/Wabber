"""Expose business-only tool functions through a Wabber Runtime in either harness."""

from __future__ import annotations

import asyncio
import inspect
from functools import wraps
from typing import get_type_hints
from uuid import uuid4

from atsuite.runtime import InvocationRequest, RuntimeAdapter


class RuntimeToolset:
    """Own one Runtime session and generate native framework tools from business signatures."""

    def __init__(self, runtime: RuntimeAdapter, target_id: str, *, timeout: float = 240):
        self.runtime = runtime
        self.target_id = target_id
        self.timeout = timeout
        self.uid = uuid4().hex
        self.session = None

    async def __aenter__(self):
        opening = asyncio.create_task(
            asyncio.to_thread(
                self.runtime.open_session,
                self.runtime.targets[self.target_id],
                self.uid,
            )
        )
        try:
            self.session = await asyncio.shield(opening)
        except asyncio.CancelledError:
            self.session = await opening
            await self.__aexit__()
            raise
        return self

    async def __aexit__(self, *exc):
        try:
            await asyncio.to_thread(self.runtime.close_session, self.session)
            self.session = None
        finally:
            close = getattr(self.runtime, "close", None)
            if close:
                await asyncio.to_thread(close)

    def _proxy(self, tool):
        function = getattr(tool, "func", tool)
        name = getattr(tool, "name", function.__name__)

        @wraps(function)
        async def invoke(*positional, **args):
            if self.session is None:
                raise RuntimeError("Open RuntimeToolset with 'async with' before invoking tools")
            if "uid" in args or any(key.startswith("__atsuite_") for key in args):
                raise ValueError("Runtime identity and snapshot arguments are managed by Wabber")
            args = public_signature.bind(*positional, **args).arguments
            pending = asyncio.create_task(
                asyncio.to_thread(
                    self.runtime.invoke,
                    InvocationRequest(
                        target_id=self.target_id,
                        tool_name=name,
                        args=args,
                        uid=self.uid,
                        session=self.session,
                        call_id=uuid4().hex,
                        timeout=self.timeout,
                    ),
                )
            )
            try:
                response = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # A worker thread cannot be cancelled; drain its bounded RPC before releasing the Session.
                try:
                    await pending
                except Exception:
                    pass
                raise
            if response.status != "ok":
                raise RuntimeError(response.error or response.status)
            return response.result

        signature = inspect.signature(function)
        hints = get_type_hints(function, include_extras=True)
        public_signature = signature.replace(
            parameters=[
                p.replace(annotation=hints.get(p.name, p.annotation))
                for p in signature.parameters.values()
                if p.name != "uid"
            ],
            return_annotation=hints.get("return", signature.return_annotation),
        )
        invoke.__signature__ = public_signature
        invoke.__annotations__ = {key: value for key, value in hints.items() if key != "uid"}
        invoke.__name__ = name
        invoke.__doc__ = inspect.getdoc(function) or getattr(tool, "docstring", None) or name
        return invoke

    def langchain_tools(self, tools) -> list:
        """Build native LangChain tools; their business functions execute in the Runtime."""
        from langchain_core.tools import StructuredTool

        return [StructuredTool.from_function(coroutine=self._proxy(tool)) for tool in tools]

    def openai_tools(self, tools) -> list:
        """Build validated SDK function tools, preserving the business argument schema."""
        from agents import function_tool

        return [function_tool(self._proxy(tool), strict_mode=False, failure_error_function=None) for tool in tools]
