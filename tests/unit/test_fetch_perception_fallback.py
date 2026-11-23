"""Search-call failures advance native fallbacks without dispatching robot work."""
import json
from unittest.mock import Mock

import py_trees
import pytest

from src.core.action import ActionResult
from src.core.failure import FailureCode, FailureInfo, FailureOrigin
from src.core.llm_planner import LLMPlanner
from src.plugin.actions.fetch import FetchObject
from src.plugin.actions.observe import ObserveScene, ScanScene, _DetectTargetInView
from tests.unit.test_workflow_catalog import make_context


@pytest.mark.parametrize("phase", ["ScanScene", "ObserveScene"])
@pytest.mark.parametrize("code", [FailureCode.NETWORK_ERROR, FailureCode.INVALID_STATE,
                                 FailureCode.INVALID_REQUEST, FailureCode.UNKNOWN,
                                 FailureCode.CANCELLED])
def test_fetch_skips_failed_perception_call_only(monkeypatch, phase, code, native_scan=False):
    from src.plugin.actions.navigation import MoveToTarget
    from src.plugin.actions.grasp_alignment import EstimateSimplePcaGrasp
    from src.plugin.actions.hsr_pickup import HsrPickup
    from src.plugin.actions.manipulation import GraspObjectAction

    ctx = make_context()
    trace, retained_highlights = [], []
    failed = False
    ctx.request_user_input = Mock(return_value="table_1")
    ctx.modules.call = Mock(return_value={"content": json.dumps({
        "known_object_ids": ["jar_near"], "additional_furniture_ids": ["table_1"]})})
    monkeypatch.setattr(LLMPlanner, "_resolve_llm_module_name", lambda self: "planner_llm")

    def immediately(work, *args, callback=None, error_callback=None, **kwargs):
        callback(work(*args, **kwargs))
    ctx.run_in_context_loop = immediately

    class Operation(py_trees.behaviour.Behaviour):
        def __init__(self, action, name):
            super().__init__(name)
            self.action, self.result = action, None

        def update(self):
            nonlocal failed
            kind = type(self.action).__name__
            trace.append((kind, getattr(self.action, "target", None)))
            if kind == "ScanScene":
                assert not self.action.strict_failures
            if kind == "MoveToTarget" and self.action.target == "table_1" and not any(k == "HsrPickup" for k, _ in trace):
                retained_highlights.extend(ctx.world_state.get("scene_known_objects") or [])
            if kind == phase and not failed:
                failed = True
                self.result = ActionResult(status="failure", reason="original call error",
                    failure=FailureInfo(code, origin=FailureOrigin.PERCEPTION))
                return py_trees.common.Status.FAILURE
            self.result = ActionResult(status="success")
            return py_trees.common.Status.SUCCESS

    for cls in (MoveToTarget, ScanScene, ObserveScene, EstimateSimplePcaGrasp, HsrPickup, GraspObjectAction):
        if native_scan and cls is ScanScene:
            continue
        monkeypatch.setattr(cls, "build_node", lambda self, context, name: Operation(self, name))
    if native_scan:
        ctx.head = None
        ctx.sensor_data_center = Mock()
        outcomes = iter([code, FailureCode.NOT_FOUND, FailureCode.NOT_FOUND, None])

        class Detector(py_trees.behaviour.Behaviour):
            def update(self):
                outcome = next(outcomes)
                if self.name == "Detect 0":
                    trace.append(("ScanScene", "blue jar"))
                self.result = ActionResult(status="failure" if outcome else "success",
                    failure=FailureInfo(outcome, origin=FailureOrigin.PERCEPTION) if outcome else None)
                return py_trees.common.Status.FAILURE if outcome else py_trees.common.Status.SUCCESS
        monkeypatch.setattr(_DetectTargetInView, "build_node", lambda self, context, name: Detector(name))
    action = FetchObject("blue jar")
    root = action.build_node(ctx, "Fetch")
    for _ in range(20):
        root.tick_once()
        if root.status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            break
    kinds = [kind for kind, _ in trace]
    assert ctx.execution.control is None
    if code == FailureCode.CANCELLED:
        assert root.status == py_trees.common.Status.FAILURE
        assert kinds.count("MoveToTarget") == 1
        assert "HsrPickup" not in kinds
        ctx.request_user_input.assert_not_called()
    else:
        assert root.status == py_trees.common.Status.SUCCESS
        assert kinds.count(phase) == 2
        assert kinds.count("HsrPickup") == 1
        assert {row["id"] for row in retained_highlights} == {"jar_near"}
        assert action._fatal is None
        # The failed visit never reaches grasp estimation/pickup.
        assert trace.index(("MoveToTarget", "table_1")) < kinds.index("EstimateSimplePcaGrasp")
        ctx.request_user_input.assert_called_once()  # Delivery question only.


def test_fetch_reads_aggregate_scan_error_instead_of_last_view_miss(monkeypatch):
    test_fetch_skips_failed_perception_call_only(monkeypatch, "ScanScene", FailureCode.UNKNOWN, native_scan=True)


@pytest.mark.parametrize("outcomes,expected,visited", [
    ([FailureCode.NETWORK_ERROR, None], None, 2),
    ([FailureCode.NETWORK_ERROR, FailureCode.NOT_FOUND, FailureCode.NOT_FOUND], FailureCode.NETWORK_ERROR, 3),
    ([FailureCode.UNKNOWN, FailureCode.NOT_FOUND, FailureCode.NOT_FOUND], FailureCode.UNKNOWN, 3),
    ([FailureCode.CANCELLED, None], FailureCode.CANCELLED, 1),
])
def test_non_strict_scan_preserves_errors_and_can_recover(monkeypatch, outcomes, expected, visited):
    ctx = make_context()
    ctx.head = None
    ctx.sensor_data_center = Mock()
    calls = []
    speech = Mock()
    monkeypatch.setattr("src.core.logger.speak", speech)

    class Detect(py_trees.behaviour.Behaviour):
        def update(self):
            code = outcomes[len(calls)]
            calls.append(self.name)
            self.result = ActionResult(status="failure" if code else "success", reason="detector error" if code else None,
                failure=FailureInfo(code, origin=FailureOrigin.PERCEPTION) if code else None)
            return py_trees.common.Status.FAILURE if code else py_trees.common.Status.SUCCESS
    monkeypatch.setattr(_DetectTargetInView, "build_node", lambda self, context, name: Detect(name))
    root = ScanScene("jar", exploration_mode=True, strict_failures=False).build_node(ctx, "Scan")
    root.tick_once()
    assert len(calls) == visited
    assert root.result.status == ("failure" if expected else "success")
    if expected:
        assert root.result.failure.code == expected
        assert root.result.reason == "detector error"
    assert not any(call.args[0] == "%s not found" for call in speech.call_args_list)


@pytest.mark.parametrize("content", ["not json", "{}", "[]", '{"status":"Found","mask":[]}'])
def test_invalid_observation_response_is_not_object_absence(content):
    action = ObserveScene("jar", replan_on_failure=False)
    with pytest.raises(ValueError, match="invalid_model_response"):
        action._extract_bb_and_mask(content)
    ctx = make_context()
    action._context = ctx
    action._extract_content = Mock(return_value=content)
    action.response_callback({})
    result = action.check_result(ctx)
    assert result.status == "failure"
    assert result.failure.code == FailureCode.UNKNOWN


def test_explicit_not_found_remains_not_found():
    assert ObserveScene("jar")._extract_bb_and_mask('{"status":"Not Found"}') == (None, None)
