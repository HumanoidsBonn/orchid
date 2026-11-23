"""Tests for LLM chat module and planner integration."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import numpy as np

from src.core.context import Context
from src.core.controllers import Pose3D
from src.core.llm_planner import LLMPlanner, PlanGenerationError
from src.core.manifest import ModuleEndpoint
from src.core.module_proxy import ModuleProxy
from src.core.modules.llm_chat_module import LLMChatConfig, LLMChatModule
from src.core.world_state import WorldState
from src.tree.plan import ActionNode, Plan


DEEPSEEK_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {
                        "goal": "go kitchen",
                        "root": {
                            "type": "action",
                            "name": "navigate_to",
                            "action_type": "navigation",
                            "params": {"target": "kitchen"},
                        },
                    }
                )
            }
        }
    ]
}

LOCAL_RESPONSE = {
    "choices": [{"message": {"content": '{"goal":"pick cup","root":{"type":"sequence","name":"main","children":[]}}'}}]
}


class TestLLMChatConfig:
    def test_deepseek_preset(self):
        cfg = LLMChatConfig.from_endpoint_config({"provider": "deepseek"})
        assert cfg.provider == "deepseek"
        assert cfg.model == "deepseek-chat"
        assert cfg.requires_api_key is True
        assert cfg.api_key_env == "DEEPSEEK_API_KEY"

    def test_local_ollama_preset(self):
        cfg = LLMChatConfig.from_endpoint_config(
            {"provider": "ollama", "model": "mistral"}
        )
        assert cfg.provider == "ollama"
        assert cfg.model == "mistral"
        assert cfg.requires_api_key is False


class TestLLMChatModule:
    def test_call_sync_builds_openai_request(self):
        module = LLMChatModule(
            "https://api.deepseek.com/chat/completions",
            {"provider": "deepseek", "api_key": "test-key"},
        )

        with patch.object(module._http, "call_sync", return_value=DEEPSEEK_RESPONSE) as mock_post:
            result = module.call_sync(
                {"system_prompt": "You are a planner", "prompt": "go kitchen"}
            )

        assert "go kitchen" in result["content"]
        mock_post.assert_called_once()
        body, kwargs = mock_post.call_args.args[0], mock_post.call_args.kwargs
        assert body["model"] == "deepseek-chat"
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["content"] == "go kitchen"

    def test_local_lan_endpoint_without_api_key(self):
        module = LLMChatModule(
            "http://192.168.1.50:11434/v1/chat/completions",
            {"provider": "ollama", "model": "llama3.2"},
        )

        with patch.object(module._http, "call_sync", return_value=LOCAL_RESPONSE) as mock_post:
            result = module.call_sync({"prompt": "pick cup"})

        assert "pick cup" in result["content"]
        body = mock_post.call_args.args[0]
        assert body["model"] == "llama3.2"

    def test_call_sync_includes_max_tokens_from_manifest(self):
        module = LLMChatModule(
            "http://192.168.1.50:11434/v1/chat/completions",
            {"provider": "ollama", "model": "llama3.2", "max_tokens": 256},
        )

        with patch.object(module._http, "call_sync", return_value=LOCAL_RESPONSE) as mock_post:
            result = module.call_sync({"prompt": "pick cup"})

        assert "pick cup" in result["content"]
        body = mock_post.call_args.args[0]
        assert body["max_tokens"] == 256


class TestLLMPlanner:
    @pytest.fixture
    def planner_context(self) -> Context:
        modules = ModuleProxy(
            [
                ModuleEndpoint(
                    name="planner_llm",
                    endpoint="https://api.deepseek.com/chat/completions",
                    type="llm",
                    config={"provider": "deepseek", "api_key": "test-key"},
                )
            ]
        )
        return Context(
            robot_id="test_bot",
            hardware={"camera": "test"},
            modules=modules,
            supported_actions={"navigation": object},
            world_state=WorldState(max_history=5),
        )

    def test_generate_plan_via_module_proxy(self, planner_context):
        planner = LLMPlanner(planner_context)
        llm_module = planner_context.modules._get_or_create_module("planner_llm")

        with patch.object(
            llm_module, "call_sync", return_value={"content": DEEPSEEK_RESPONSE["choices"][0]["message"]["content"]}
        ):
            plan = planner.generate_plan("go to the kitchen")

        assert plan.goal == "go kitchen"
        assert isinstance(plan.root, ActionNode)
        assert plan.root.name == "navigate_to"

    def test_generate_plan_keeps_user_input_as_ask_user_action(self, planner_context):
        planner = LLMPlanner(planner_context)
        llm_module = planner_context.modules._get_or_create_module("planner_llm")

        with patch.object(
            llm_module,
            "call_sync",
            return_value={
                "content": json.dumps(
                    {
                        "goal": "find the cup",
                        "root": {
                            "type": "action",
                            "name": "ask_user",
                            "action_type": "AskUser",
                            "params": {"prompt": "Where should I look?"},
                        },
                    }
                )
            },
        ):
            plan = planner.generate_plan("find the cup")

        assert isinstance(plan.root, ActionNode)
        assert plan.root.action_type == "AskUser"
        assert plan.root.params["prompt"] == "Where should I look?"

    def test_generate_plan_accepts_markdown_fenced_json(self, planner_context):
        planner = LLMPlanner(planner_context)
        llm_module = planner_context.modules._get_or_create_module("planner_llm")
        content = """```json
{
  "goal": "go kitchen",
  "root": {
    "type": "action",
    "name": "navigate_to",
    "action_type": "navigation",
    "params": {"target": "kitchen"}
  }
}
```"""

        with patch.object(llm_module, "call_sync", return_value={"content": content}):
            plan = planner.generate_plan("go to the kitchen")

        assert plan.goal == "go kitchen"
        assert isinstance(plan.root, ActionNode)
        assert plan.root.params["target"] == "kitchen"

    def test_load_response_json_accepts_fenced_nested_response(self, planner_context):
        planner = LLMPlanner(planner_context)
        response = json.dumps(
            {
                "response": """```json
{"goal":"pick cup","root":{"type":"sequence","name":"main","children":[]}}
```"""
            }
        )

        data = planner._load_response_json(response)

        assert data["goal"] == "pick cup"
        assert data["root"]["type"] == "sequence"

    def test_load_response_json_repairs_unmatched_closing_bracket(self, planner_context):
        planner = LLMPlanner(planner_context)
        response = (
            '{"goal":"bring me the blue jar",'
            '"root":{"type":"sequence","name":"main","children":['
            '{"type":"action","name":"pick","action_type":"PickTarget","params":{"target":"blue_jar"}}'
            ']}}]'
        )

        data = planner._load_response_json(response)

        assert data["goal"] == "bring me the blue jar"
        assert data["root"]["children"][0]["action_type"] == "PickTarget"

    def test_prompt_includes_current_task_action_history(self, planner_context):
        planner = LLMPlanner(planner_context)
        prompt = planner._build_user_prompt(
            "pick the cup",
            action_history=[
                {
                    "action_name": "MoveToTarget",
                    "params": {"target": "table"},
                    "result": {"status": "success", "reason": None},
                }
            ],
        )

        assert "current_task_progress" in prompt
        assert "continue from the robot/world state" in prompt
        assert "completed_planner_actions" in prompt
        assert "MoveToTarget" in prompt

    def test_prompt_sanitizes_action_history_runtime_objects(self, planner_context):
        planner = LLMPlanner(planner_context)
        prompt = planner._build_user_prompt(
            "continue pickup",
            action_history=[
                {
                    "action_name": "PickAtPose",
                    "action_class": "PickAtPose",
                    "action_type": "PickAtPose",
                    "planner_visible": True,
                    "params": {
                        "pose": Pose3D(x=0.1, y=0.2, z=0.3, frame_id="odom"),
                    },
                    "result": {
                        "status": "failure",
                        "reason": "test",
                        "payload": {"values": np.array([1, 2])},
                        "suggested_replan_context": {
                            "pose": Pose3D(x=1.0, y=2.0, z=3.0),
                        },
                    },
                }
            ],
            last_action_result={
                "action_name": "PickAtPose",
                "status": "failure",
                "reason": "test",
                "suggested_replan_context": {
                    "pose": Pose3D(x=1.0, y=2.0, z=3.0),
                },
            },
        )

        assert '"frame_id": "odom"' in prompt
        assert '"values": [' not in prompt
        assert "Object of type Pose3D" not in prompt

    def test_prompt_excludes_large_runtime_metadata(self, planner_context):
        planner = LLMPlanner(planner_context)
        prompt = planner._build_user_prompt(
            "continue pickup",
            action_history=[
                {
                    "action_name": "MoveToManipulationPose",
                    "action_class": "MoveToManipulationPose",
                    "action_type": "MoveToManipulationPose",
                    "planner_visible": True,
                    "params": {"target": "cup"},
                    "result": {
                        "status": "needs_replan",
                        "reason": "no_ik_feasible_candidate",
                        "payload": {"raw_result": "x" * 10000},
                        "suggested_replan_context": {
                            "selection_summary": {
                                "nav_feasible_candidates": 4,
                                "ik_feasible_candidates": 0,
                            }
                        },
                    },
                }
            ],
        )

        assert "selection_summary" in prompt
        assert "nav_feasible_candidates" in prompt
        assert "raw_result" not in prompt

    def test_missing_llm_module_raises(self):
        planner = LLMPlanner(Context(modules=ModuleProxy([])))
        with pytest.raises(PlanGenerationError, match="No LLM module configured"):
            planner.generate_plan("do something")

    def test_module_error_surfaces_as_plan_generation_error(self, planner_context):
        planner = LLMPlanner(planner_context)
        llm_module = planner_context.modules._get_or_create_module("planner_llm")

        with patch.object(llm_module, "call_sync", return_value={"error": "HTTP 401"}):
            with pytest.raises(PlanGenerationError, match="HTTP 401"):
                planner.generate_plan("go kitchen")
