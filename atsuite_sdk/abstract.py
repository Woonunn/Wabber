import inspect
import asyncio
from typing import Any, Callable, Dict, Optional, Type, get_type_hints
from pydantic import BaseModel, create_model
from functools import wraps
from atsuite_sdk.resources import observe_call

class UnifiedAgentTool:
    """
    统一 Agent Tool 包装器。
    用于封装 Agent Tool 逻辑，并自动提取元数据以适配 MCP 或 FaaS。
    """
    def __init__(self, func: Callable, name: Optional[str] = None, *, stateful: bool = False, access: Optional[str] = None, domain: str = ""):
        self.func = func
        self.name = name or func.__name__
        from atsuite_sdk.state import load_manifest
        configured = load_manifest().get("tool_access", {}).get(self.name, {})
        if configured.get("access") != "stateless":
            access = configured.get("access", access)
        domain = configured.get("domain", domain)
        self.access = access if access is not None else ("rw" if stateful else "stateless")
        if self.access not in {"r", "w", "rw", "stateless"}:
            raise ValueError(f"Invalid Tool access: {self.access}")
        self.stateful = self.access != "stateless"
        self.domain = domain
        self.docstring = inspect.getdoc(func) or "No description provided."
        self.parameters_schema = self._generate_schema()

        
    def _generate_schema(self) -> Dict[str, Any]:
        """
        利用 Pydantic 动态创建模型，从而生成符合 OpenAPI/JSON Schema 标准的参数描述。
        """
        type_hints = get_type_hints(self.func)
        fields = {}
        
        # 移除返回值类型的注解，只保留参数
        if 'return' in type_hints:
            del type_hints['return']
            
        # 获取默认值
        signature = inspect.signature(self.func)
        
        for param_name, param_type in type_hints.items():
            # 确保参数存在于 signature.parameters 中（排除 'return'）
            if param_name not in signature.parameters:
                continue
            default_value = signature.parameters[param_name].default
            if default_value == inspect.Parameter.empty:
                fields[param_name] = (param_type, ...)
            else:
                fields[param_name] = (param_type, default_value)

        if "uid" not in fields:
            fields["uid"] = (Optional[str], None)
                
        # 动态创建 Pydantic 模型
        DynamicModel = create_model(f"{self.name}_Args", **fields)
        return DynamicModel.model_json_schema()

    def __call__(self, *args, **kwargs):
        """保留原始 Agent Tool 的调用方式"""
        return self.func(*args, **kwargs)

    def to_mcp_tool(self) -> Dict[str, Any]:
        """导出为 MCP Tool 定义格式"""
        return {
            "name": self.name,
            "description": self.docstring,
            "inputSchema": self.parameters_schema
        }

    @observe_call()
    def execute_from_dict(self, args_dict: Dict[str, Any]) -> Any:
        """通用执行入口，接受字典参数（适用于 MCP 和 FaaS 的 JSON Body）"""
        call_args = dict(args_dict)
        snapshot_payload = call_args.pop("__atsuite_state_snapshot", None)
        from atsuite_sdk.state import get_state_runtime

        runtime = get_state_runtime()
        uid = call_args.get("uid")
        if "uid" not in inspect.signature(self.func).parameters:
            call_args.pop("uid", None)
        with runtime.execution(self.func.__module__, uid, self.access, self.domain, snapshot_payload):
            result = self.func(**call_args)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            if hasattr(result, "to_dict"):
                result = result.to_dict(orient="records")
            return result

# --- 装饰器 ---
class AgentToolRegistry:
    def __init__(self):
        self.functions: Dict[str, UnifiedAgentTool] = {}

    def tool(self, name: Optional[str] = None, *, stateful: bool = False, access: Optional[str] = None, domain: str = ""):
        def decorator(func):
            unified = UnifiedAgentTool(func, name, stateful=stateful, access=access, domain=domain)
            self.functions[unified.name] = unified
            return unified
        return decorator

# 全局注册表实例
registry = AgentToolRegistry()
