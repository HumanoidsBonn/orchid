from __future__ import annotations

import json
from typing import Any

from src.core.context import Context
from src.core.module_proxy import ModuleProxy
from src.core.serialization import to_json_safe
from src.tree.plan import Plan
import src.core.logger as log

DEFAULT_LLM_MODULE = "planner_llm"

SYSTEM_PROMPT = """You are a robot task planner. You will be provided the environment state in the form of a scene graph and the robot's
capability list that mentions the kind of control that is available. Given a natural-language task produce a ordered task plan as JSON which will be used to create a behavior tree.

Rules:
SCENE_GRAPH_CHECKS:
- If scene graph entry is empty or null, then be aware that no environment information is available and directly say you cannot do it.
- Use context about other possible locations around only if scene is not empty.
- Based on the scene graph, try to understand the kind of environment the robot is in like a warehouse, a home, a lab, etc. and then think of the kind of task plan that is appropriate for the environment.

- A scene-graph object is a belief, not a fact: Always first only confirm the particular location with Perception actions. After confirming, if the relevant object wasn't available you can AskUser or Explore the remaining likely locations in the SceneGraph based on action history. 
- For underspecified commands (e.g., “bring me something to eat/drink”), create one fallback tree containing: first, a separate attempt for every task-relevant scene-graph object; then, a separate inspection branch for top (<5) likely reachable furniture node where the target object could be found, ordered by how likely it is to contain or support the requested item based on its label and supported objects. Use exact furniture IDs, exclude room nodes and furniture marked UNREACHABLE, and include all branches in the initial plan.
- If the previous plan stopped because InspectScene reported a found_label, the search is over: plan only the remaining task for that found_label (use it verbatim as the target; always do ObserveScene again in the replan first). Do not re-emit the search fallback.

- Make use of TYPES_OF_NODES_AVAILABLE mentioned later to achieve the flow.
- Scene-graph nodes have exact IDs such as desk_5 and cup_22.
- Use exact node IDs in action parameters.
- Objects may be supported by furniture.
- To approach an object, MoveToTarget will use the supporting furniture's
  access poses.
- Never invent an object ID.
- If multiple nodes share a label, select one using room/task context.
- If selection is impossible, request clarification via AskUser action.
PLANNING_CHECKS:
- Use ONLY actions from the provided capability list (via action_type).
- Do NOT invent actions the robot does not support.
ERROR_CHECKS:
- If the task is not possible due to - lack of required hardware or action capabilities, report an error saying I cannot do this task.
- Use action node with "error" as the action type for the event that the task is not possible due to above case only and include the reason as well.
- Do not return any other plan with any other JSON objects in this case.
- Do not raise an error flag if the location of the object of interest is not known, directly start looking around with fallback node.
ASK_USER_INPUT:
- Do not use the AskUser action at the start itself if the location of the object of interest is not known in the scene graph.
- First start looking around with fallback node and at the end add the AskUser action node, dont ask this as the first node at all.
- If the task is to get/ bring something on success always ask at the end, where?
TYPES_OF_NODES_AVAILABLE:
- Use "sequence" for ordered steps that must all succeed in order.
- Use "fallback" (or "selector") for alternatives: try A, if it fails try B.
- Use "parallel" only when independent actions may run concurrently.
- Use "retry" decorator when an action may need multiple attempts.
- Use "timeout" decorator to bound long-running actions.
- Use "condition" nodes for state checks; use "action" nodes for execution.
- Every action node MUST set action_type to a supported capability key.
- Keep plans minimal: no unnecessary nesting or redundant steps.
- Output ONLY valid JSON matching the schema. No markdown, no explanation."""

PLAN_JSON_SCHEMA = """
Top-level object:
  {"goal": "<original task>", "root": <PlanNode>}

PlanNode forms:
  sequence/fallback/parallel/selector:
    {"type": "sequence"|"fallback"|"parallel"|"selector", "name": str, "children": [<PlanNode>, ...]}
  action:
    {"type": "action", "name": str, "action_type": "<capability_key>", "params": {...}}
  retry:
    {"type": "retry", "name": str, "params": {"num_attempts": int}, "child": <PlanNode>}
  timeout:
    {"type": "timeout", "name": str, "params": {"duration": float}, "child": <PlanNode>}
"""

# PlanNode forms:
#   sequence/fallback/parallel/selector:
#     {"type": "sequence"|"fallback"|"parallel"|"selector", "name": str, "children": [<PlanNode>, ...]}
#   action:
#     {"type": "action", "name": str, "action_type": "<capability_key>", "params": {...}}
#   condition:
#     {"type": "condition", "name": str, "params": {...}}
#   retry:
#     {"type": "retry", "name": str, "params": {"num_attempts": int}, "child": <PlanNode>}
#   timeout:
#     {"type": "timeout", "name": str, "params": {"duration": float}, "child": <PlanNode>}

class PlanGenerationError(RuntimeError):
    """Raised when the LLM response cannot be converted into a valid Plan."""


class UserInputRequired(RuntimeError):
    """Raised when the LLM requests clarification from the user."""


class LLMPlanner:
    """Convert natural-language commands into a :class:`~src.tree.plan.Plan`."""

    def __init__(self, context: Context, llm_module: str | None = None):
        self.context = context
        self.llm_module = llm_module

    def generate_plan(
        self,
        command: str,
        user_feedback: str | None = None,
        last_action_result: dict[str, Any] | None = None,
        build_error: str | None = None,
        action_history: list[dict[str, Any]] | None = None,
    ) -> Plan:
        """Generate a :class:`Plan` from an English command."""
        module_name = self._resolve_llm_module_name()
        response = self.context.modules.call(
            module_name,
            {
                "system_prompt": SYSTEM_PROMPT,
                "prompt": self._build_user_prompt(
                    command,
                    user_feedback=user_feedback,
                    last_action_result=last_action_result,
                    build_error=build_error,
                    action_history=action_history,
                ),
            },
        )
        content = self._extract_llm_content(response, module_name)
        return self._parse_plan_json(content, command)

    def _build_user_prompt(
        self,
        command: str,
        user_feedback: str | None = None,
        last_action_result: dict[str, Any] | None = None,
        build_error: str | None = None,
        action_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build the user message with task, capabilities, scene graph, and schema."""
        from src.core.scene_graph_query import SceneGraphQueryHelper

        # capabilities = sorted(self.context.supported_actions.keys())
        capabilities = [(key, (self.context.supported_actions[key].__doc__))
                        for key in sorted(self.context.supported_actions.keys())]
        # log.info(capabilities)
        controllers = {
            name: controller is not None
            for name, controller in (
                ("nav", self.context.nav),
                ("arm", self.context.arm),
                ("head", self.context.head),
                ("gripper", self.context.gripper),
            )
        }

        # Build scene graph description
        scene_info = "No scene graph available."
        if self.context.scene_graph:
            helper = SceneGraphQueryHelper(self.context.scene_graph)
            scene_info = helper.get_scene_description()

        prompt = (
            f"Task: {command}\n\n"
            f"Robot ID: {self.context.robot_id}\n"
            f"Scene Environment:\n{scene_info}\n\n"
            f"Available action capabilities: {json.dumps(capabilities)}\n"
            f"Available conditions: {json.dumps([(name, cls.__doc__) for name, cls in sorted(self.context.supported_conditions.items())])}\n"
            f"Active controllers: {json.dumps(controllers)}\n"
            f"Robot hardware constraints: {json.dumps(self.context.hardware, indent=2)}\n\n"
            "Produce a behavior tree plan as JSON.\n"
            f"{PLAN_JSON_SCHEMA.strip()}\n\n"
            "Set goal to the original task text. Return only the JSON object."
        )

        if user_feedback:
            prompt += f"\n\nUser clarified: {user_feedback}"

        if action_history:
            action_history = to_json_safe(action_history)
            planner_visible = [
                item for item in action_history
                if item.get("planner_visible", True)
                and not str(item.get("action_class") or item.get("action_name") or "").startswith("_")
            ]
            completed = [
                {
                    "action_name": item.get("action_name"),
                    "action_type": item.get("action_type"),
                    "params": item.get("params"),
                    "duration_s": item.get("duration_s"),
                }
                for item in planner_visible
                if (item.get("result") or {}).get("status") == "success"
            ]
            boundary = next(
                (
                    {
                        "action_name": item.get("action_name"),
                        "action_type": item.get("action_type"),
                        "params": item.get("params"),
                        "status": (item.get("result") or {}).get("status"),
                        "reason": (item.get("result") or {}).get("reason"),
                        "suggested_replan_context": (item.get("result") or {}).get("suggested_replan_context"),
                    }
                    for item in reversed(planner_visible)
                    if (item.get("result") or {}).get("status") != "success"
                ),
                None,
            )
            progress = {
                "completed_planner_actions": completed,
                "replan_boundary": boundary,
            }
            prompt += (
                "\n\ncurrent_task_progress:\n"
                "The next plan must continue from the robot/world state after the completed actions. "
                "Do not restart from the beginning unless a completed action must be repeated because state changed.\n"
                f"{json.dumps(to_json_safe(progress), indent=2, default=str)}"
            )

        if last_action_result:
            last_action_result = to_json_safe(last_action_result)
            prompt += (
                "\n\nThe previous plan execution failed because "
                f"action {last_action_result.get('action_name')} returned {last_action_result.get('status')}"
            )
            if last_action_result.get("reason"):
                prompt += f": {last_action_result['reason']}"
            if last_action_result.get("suggested_replan_context"):
                prompt += (
                    "\nUse this suggested replan context: "
                    f"{json.dumps(last_action_result['suggested_replan_context'], indent=2, default=str)}"
                )

        if build_error:
            prompt += (
                "\n\nThe previous plan could not be converted into an executable tree because: "
                f"{build_error}"
                "\nAvoid unsupported action names, decorators, or node types."
            )

        return prompt

    def _resolve_llm_module_name(self) -> str:
        """Pick the configured LLM module from the manifest."""
        modules = self.context.modules
        if not isinstance(modules, ModuleProxy):
            raise PlanGenerationError("Context has no ModuleProxy configured")

        if self.llm_module and modules.has(self.llm_module):
            return self.llm_module

        if modules.has(DEFAULT_LLM_MODULE):
            return DEFAULT_LLM_MODULE

        for name, endpoint in modules.endpoints.items():
            if endpoint.type == "llm":
                return name
            if endpoint.type == "http" and endpoint.config.get("protocol") == "openai_chat":
                return name

        raise PlanGenerationError(
            "No LLM module configured. Add a module with type 'llm' or "
            "type 'http' and config.protocol='openai_chat' to the robot manifest."
        )

    def _extract_llm_content(self, response: dict[str, Any], module_name: str) -> str:
        """Normalize module response into raw LLM text."""
        if "error" in response:
            raise PlanGenerationError(
                f"LLM module '{module_name}' failed: {response['error']}"
            )

        content = response.get("content")
        if isinstance(content, str) and content.strip():
            return content

        raw = response.get("raw")
        if isinstance(raw, dict):
            choices = raw.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                nested = message.get("content")
                if isinstance(nested, str) and nested.strip():
                    return nested

        raise PlanGenerationError(
            f"LLM module '{module_name}' returned no message content"
        )

    def _parse_plan_json(self, response: str, command: str) -> Plan:
        """Parse the LLM JSON response into the shared ``Plan`` model."""
        try:
            data = self._load_response_json(response)
            log.info("Parsed LLM response JSON: %s", json.dumps(data, indent=2))
            goal = data.get("goal", command)
            root_data = data.get("root", data)
            return Plan(goal=goal, root=root_data)
        except UserInputRequired:
            raise
        except Exception as exc:
            log.exception("Failed to parse LLM plan for command %r: %s", command, exc, 'response:', response)
            raise PlanGenerationError(
                f"Invalid plan JSON for command {command!r}: {exc}"
            ) from exc

    @staticmethod
    def _plan_contains_user_input_action(plan: Plan, prompt: str) -> bool:
        """Return True if the plan contains a user input action."""
        from src.tree.plan import ActionNode, CompositeNode, DecoratorNode

        def traverse(node):
            if isinstance(node, ActionNode):
                prompt = node.params.get("prompt", "") if node.params else "Please clarify the task."
                return node.action_type == "user_input"
            if isinstance(node, CompositeNode):
                return any(traverse(child) for child in node.children)
            if isinstance(node, DecoratorNode):
                return traverse(node.child)
            return False

        return traverse(plan.root)
    
    def _load_response_json(self, response: str) -> dict[str, Any]:
        """Parse raw LLM output, unwrapping common envelope formats."""
        text = self._strip_json_markdown_fence(response)
        data = self._loads_json_with_repair(text)
        if not isinstance(data, dict):
            raise TypeError(f"Expected JSON object, got {type(data).__name__}")

        if "root" in data:
            return data

        if "response" in data:
            nested = data["response"]
            if isinstance(nested, str):
                nested = self._loads_json_with_repair(self._strip_json_markdown_fence(nested))
            if isinstance(nested, dict):
                return nested

        if data.get("type") in {
            "sequence",
            "selector",
            "parallel",
            "fallback",
            "action",
            "condition",
            "retry",
            "timeout",
            "inverter",
            "repeat",
        }:
            return {"goal": "", "root": data}

        raise ValueError("LLM response does not contain a plan root node")

    @classmethod
    def _loads_json_with_repair(cls, text: str) -> Any:
        """Parse JSON, then try one conservative bracket-repair pass."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = cls._repair_json_closure(cls._extract_largest_json_object(text))
            if repaired == text:
                raise
            return json.loads(repaired)

    @staticmethod
    def _extract_largest_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return text
        return text[start : end + 1]

    @staticmethod
    def _repair_json_closure(text: str) -> str:
        pairs = {"{": "}", "[": "]"}
        closers = {"}", "]"}
        stack: list[str] = []
        output: list[str] = []
        in_string = False
        escaped = False
        started = False
        for char in text:
            if not started:
                if char != "{":
                    continue
                started = True
            if in_string:
                output.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                output.append(char)
                continue
            if char in pairs:
                stack.append(pairs[char])
                output.append(char)
                continue
            if char in closers:
                if stack and char == stack[-1]:
                    stack.pop()
                    output.append(char)
                continue
            output.append(char)
        if in_string:
            output.append('"')
        output.extend(reversed(stack))
        return "".join(output)

    @staticmethod
    def _strip_json_markdown_fence(response: str) -> str:
        """Return JSON text from raw LLM output, tolerating fenced code blocks."""
        text = response.strip()
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if not lines:
            return text
        first = lines[0].strip().lower()
        if first not in {"```", "```json"}:
            return text
        if len(lines) < 2 or lines[-1].strip() != "```":
            return text
        return "\n".join(lines[1:-1]).strip()
