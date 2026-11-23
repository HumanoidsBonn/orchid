"""Workflow labels are presentation data; queries retain backend/state identity."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import py_trees

from src.core.action import ActionResult
from src.core.failure import FailureCode, FailureInfo
from src.core.llm_planner import LLMPlanner
from src.plugin.actions.fetch import FetchObject, ResolveFetchLocations
from src.plugin.actions.observe import ScanScene, _DetectTargetInView
from src.workflows.catalog import WorkflowCatalog, WorkflowSettings
from tests.unit.test_workflow_catalog import make_context


QUERY = "red pringles cylinder"
LABEL = "Pringles can"


def test_catalog_request_plan_and_children_preserve_query_and_label():
    ctx = make_context()
    catalog = WorkflowCatalog(ctx, "robot", WorkflowSettings(items=[
        {"id": "pringles_can", "query": QUERY, "label": LABEL}]))
    request = catalog.validate("fetch", {"item_id": "pringles_can"})
    plan = catalog.build_plan(request)
    assert str(request) == plan.goal == "Fetch Pringles can"
    assert plan.root.name == "Fetch Pringles can"
    assert plan.root.params["target"] == QUERY
    assert plan.root.params["target_label"] == LABEL
    action = FetchObject(**plan.root.params)
    action.build_node(ctx, plan.root.name)
    labelled_types = {"ScanScene", "ObserveScene", "ResolveFetchLocations", "HsrPickup", "PlanHsrPickup"}
    children = [binding.action for binding in ctx.execution.bindings.values()
                if type(binding.action).__name__ in labelled_types]
    assert {type(child).__name__ for child in children} == labelled_types
    assert all(child.target == QUERY and child.target_label == LABEL for child in children)
    prompts = [binding.action.prompt for binding in ctx.execution.bindings.values()
               if type(binding.action).__name__ == "AskUser"]
    assert "Where should I look for Pringles can?" in prompts
    assert all(QUERY not in prompt for prompt in prompts)


def test_ranking_sends_query_but_speaks_label(monkeypatch):
    ctx = make_context()
    monkeypatch.setattr(LLMPlanner, "_resolve_llm_module_name", lambda self: "planner_llm")
    ctx.modules.call = Mock(return_value={"content": json.dumps({
        "known_object_ids": ["jar_near"], "additional_furniture_ids": []})})
    def immediately(work, callback=None, error_callback=None, **kwargs):
        try:
            result = work()
        except Exception as exc:
            error_callback(exc)
        else:
            callback(result)
    ctx.run_in_context_loop = immediately
    speech = Mock()
    monkeypatch.setattr("src.core.logger.speak", speech)
    action = ResolveFetchLocations(QUERY, "locations", target_label=LABEL)
    action.start(ctx)
    assert action.check_result(ctx).status == "success"
    assert json.loads(ctx.modules.call.call_args.args[1]["prompt"])["target"] == QUERY
    spoken = [call.args[0] % call.args[1:] if len(call.args) > 1 else call.args[0]
              for call in speech.call_args_list]
    assert any(LABEL in text for text in spoken)
    assert all(QUERY not in text for text in spoken)


def test_scan_detector_keeps_query_and_feedback_uses_label(monkeypatch):
    ctx = make_context()
    ctx.head = None
    ctx.sensor_data_center = Mock()
    targets = []
    class Miss(py_trees.behaviour.Behaviour):
        def update(self):
            self.result = ActionResult(status="failure", failure=FailureInfo(FailureCode.NOT_FOUND))
            return py_trees.common.Status.FAILURE
    def detector_node(action, context, name):
        targets.append(action.target)
        return Miss(name)
    monkeypatch.setattr(_DetectTargetInView, "build_node", detector_node)
    speech = Mock()
    monkeypatch.setattr("src.core.logger.speak", speech)
    root = ScanScene(QUERY, target_label=LABEL, exploration_mode=True).build_node(ctx, "Scan")
    root.tick_once()
    assert targets == [QUERY] * 3
    speech.assert_any_call("Looking for %s", LABEL)
    speech.assert_any_call("%s not found", LABEL)
    assert ctx.world_state.get(QUERY + "_visible") is False
    assert ctx.world_state.get(LABEL + "_visible") is None


def test_segmentation_and_grasp_captions_use_label_without_changing_identity(monkeypatch):
    from src.plugin.actions import perception_preview as preview
    ctx = make_context()
    header = NS(frame_id="odom", stamp=NS(sec=12, nanosec=0))
    info = NS(width=320, height=240, p=[200., 0, 160, 0, 0, 200., 120, 0, 0, 0, 1, 0],
              k=[0.] * 9, d=[], header=header)
    action = NS(_context=ctx, target=QUERY, target_label=LABEL,
                processed_frame=np.zeros((240, 320, 3), np.uint8),
                rgb_data=NS(data=NS(header=header)), camera_info_data=NS(data=info))
    monkeypatch.setattr(preview, "publish_image", Mock())
    text = Mock(wraps=preview.cv2.putText)
    monkeypatch.setattr(preview.cv2, "putText", text)
    preview.observation_preview(action, [20, 20, 60, 70], [], "odom")
    snapshot = ctx.world_state.get("perception_preview_observation")
    assert snapshot.target == QUERY and snapshot.target_label == LABEL
    pose = np.eye(4)
    pose[2, 3] = 1.
    preview.grasp_image(snapshot, pose, .08)
    captions = [call.args[1] for call in text.call_args_list]
    assert captions[0] == LABEL and captions[1].startswith(LABEL + " | opening")
    assert all(QUERY not in caption for caption in captions)
