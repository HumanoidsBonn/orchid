"""Unit tests for InspectScene's verify/retry flow with stubbed detectors."""

from __future__ import annotations

import pytest

from tests.fixtures.native_tree import tick_action, action_node
from src.core.failure import FailureInfo, FailureCode
from src.core.action import ActionResult, AsyncAction
from src.core.context import Context
from src.core.module_proxy import ModuleProxy
from src.core.world_state import WorldState
import src.plugin.actions.inspect as inspect_scene
from src.plugin.actions.inspect import InspectScene, _box_iou, _as_pixel_box


class _FakeDetector(AsyncAction):
    """Stands in for _DetectDescribedTargetInView: returns a scripted answer."""

    def __init__(self, answer, **params):
        super().__init__()
        self.index = params["index"]
        self.attempt = 0
        self.rejected_labels = []
        self.rejected_box = None
        self._answers = answer
        self._answer = None
        self.found_label = None
        self.found_box = None
        self.found_confidence = None

    def start(self, context):
        self._answer = self._answers(self)

    def check_result(self, context):
        if self._answer is None:
            return ActionResult(status="failure", reason="object_not_found")
        label, box = self._answer
        self.found_label = label
        self.found_box = box
        return ActionResult(status="success")

    def cancel(self, context):
        self._cancel_requested = True


class _FakeObserve(AsyncAction):
    """Stands in for ObserveScene: scripted per-label outcome."""

    outcomes: dict[str, object] = {}
    calls: list[dict] = []

    def __init__(self, target="", **params):
        super().__init__()
        self.target = target
        self.add_other_objects_to_sg = params.get("add_other_objects_to_sg", False)
        self.replan_on_failure = params.get("replan_on_failure", False)

    def start(self, context):
        _FakeObserve.calls.append(dict(target=self.target, add_other_objects_to_sg=self.add_other_objects_to_sg, replan_on_failure=self.replan_on_failure))

    def check_result(self, context):
        outcome = _FakeObserve.outcomes.get(self.target, "object_not_found")
        if isinstance(outcome, list):  # a SAM3 box -> success
            context.world_state.set(f"{self.target}_normalized_bbox", outcome)
            context.world_state.set(f"{self.target}_object_pcd", "pcd")
            return ActionResult(status="success")
        return ActionResult(status="failure", reason=str(outcome), failure=FailureInfo(FailureCode.NOT_FOUND if outcome == "object_not_found" else FailureCode.UNKNOWN))

    def cancel(self, context):
        self._cancel_requested = True


def _run(action, context, max_ticks=50):
    result = tick_action(action, context)
    for _ in range(max_ticks - 1):
        if result.status != "running":
            break
        result = tick_action(action, context)
    return result


@pytest.fixture
def ctx() -> Context:
    return Context(
        robot_id="test_bot",
        hardware={},
        modules=ModuleProxy([]),
        sensor_data_center=object(),
        world_state=WorldState(max_history=50),
    )


@pytest.fixture
def scripted(monkeypatch):
    """Install fakes. Returns a setter for per-view VLM answers (list per view,
    consumed one per attempt) and per-label ObserveScene outcomes."""
    _FakeObserve.outcomes = {}
    _FakeObserve.calls = []
    monkeypatch.setattr(inspect_scene, "ObserveScene", _FakeObserve)

    answers: dict[int, list] = {}
    detector_calls: list[dict] = []

    def _new_detector(self):
        def answer(detector):
            queue = answers.setdefault(int(self.index), [])
            detector_calls.append({"index": int(self.index), "attempt": detector.attempt, "rejected": list(detector.rejected_labels)})
            return queue.pop(0) if queue else None
        return _FakeDetector(answer, index=self.index)

    monkeypatch.setattr(inspect_scene._ExploreView, "_new_detector", _new_detector)

    def configure(vlm=None, observe=None):
        answers.clear()
        answers.update(vlm or {})
        _FakeObserve.outcomes = dict(observe or {})

    configure.detector_calls = detector_calls
    return configure


class TestBoxHelpers:
    def test_normalized_box_scaled_to_pixels(self):
        assert _as_pixel_box([0.5, 0.5, 1.0, 1.0]) == [240.0, 320.0, 480.0, 640.0]

    def test_pixel_box_passthrough_and_bad_input(self):
        assert _as_pixel_box([10, 20, 30, 40]) == [10.0, 20.0, 30.0, 40.0]
        assert _as_pixel_box(None) is None
        assert _as_pixel_box([1, 2, 3]) is None

    def test_iou(self):
        assert _box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
        assert _box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


class TestInspectSceneVerification:
    def test_verified_find_is_reported_as_observed(self, ctx, scripted):
        scripted(
            vlm={0: [("pringles", [100, 100, 200, 200])]},
            observe={"pringles": [110, 105, 205, 195]},
        )
        result = _run(InspectScene(target="something to eat", exploration_mode=True), ctx)

        assert result.status == "needs_replan"
        assert result.force_replan is True
        rc = result.suggested_replan_context
        assert rc["found_label"] == "pringles"
        assert rc["observed"] is True
        assert ctx.world_state.get("something to eat_scan_label") == "pringles"
        assert ctx.world_state.get("pringles_visible") is True
        assert _FakeObserve.calls[0]["replan_on_failure"] is False
        assert _FakeObserve.calls[0]["add_other_objects_to_sg"] is False

    def test_hallucination_retried_once_then_next_view(self, ctx, scripted):
        # View 0: VLM says sandwich twice, SAM3 never finds it -> exhaust retry.
        # View 1: VLM says apple, SAM3 confirms.
        scripted(
            vlm={
                0: [("sandwich", [10, 10, 50, 50]), ("sandwich", [10, 10, 50, 50])],
                1: [("apple", [100, 100, 150, 150])],
            },
            observe={"apple": [100, 100, 150, 150]},
        )
        result = _run(InspectScene(target="something to eat", exploration_mode=True, max_retries=1), ctx)

        assert result.status == "needs_replan"
        assert result.suggested_replan_context["found_label"] == "apple"
        calls = scripted.detector_calls
        # view 0 attempt 0, view 0 attempt 1 (with feedback), view 1 attempt 0
        assert [(c["index"], c["attempt"]) for c in calls] == [(0, 0), (0, 1), (1, 0)]
        assert calls[1]["rejected"] == ["sandwich"]
        # sandwich was only sent to ObserveScene once: the repeated label on the
        # retry is rejected without another SAM3 call.
        assert [c["target"] for c in _FakeObserve.calls] == ["sandwich", "apple"]

    def test_retry_accepts_corrected_label(self, ctx, scripted):
        scripted(
            vlm={0: [("sandwich", [10, 10, 50, 50]), ("banana", [10, 10, 50, 50])]},
            observe={"banana": [12, 8, 48, 52]},
        )
        result = _run(InspectScene(target="something to eat", exploration_mode=True), ctx)
        assert result.status == "needs_replan"
        assert result.suggested_replan_context["found_label"] == "banana"

    def test_max_retries_parameter_respected(self, ctx, scripted):
        scripted(vlm={0: [("a", [0, 0, 1, 1]), ("b", [0, 0, 1, 1]), ("c", [0, 0, 1, 1]), ("d", [0, 0, 1, 1])]})
        _run(InspectScene(target="x", exploration_mode=True, pan_angles=[0.0], max_retries=3), ctx)
        assert [c["attempt"] for c in scripted.detector_calls] == [0, 1, 2, 3]

    def test_zero_retries_moves_on_immediately(self, ctx, scripted):
        scripted(vlm={0: [("a", [0, 0, 1, 1]), ("b", [0, 0, 1, 1])]})
        result = _run(InspectScene(target="x", exploration_mode=True, pan_angles=[0.0], max_retries=0), ctx)
        assert result.status == "needs_replan"
        assert result.suggested_replan_context["found"] is False
        assert len(scripted.detector_calls) == 1

    def test_sam3_find_is_enough_even_if_boxes_disagree(self, ctx, scripted):
        # SAM3 finds "cup" far from where the VLM claimed; SAM3's mask is what
        # gets picked, so the find still stands.
        scripted(
            vlm={0: [("cup", [0, 0, 20, 20])]},
            observe={"cup": [400, 600, 470, 630]},
        )
        result = _run(InspectScene(target="x", exploration_mode=True, pan_angles=[0.0], max_retries=0), ctx)
        assert result.status == "needs_replan"
        assert result.suggested_replan_context["found_label"] == "cup"
        assert ctx.world_state.get("cup_object_pcd") == "pcd"

    def test_infrastructure_failure_does_not_retry_vlm(self, ctx, scripted):
        scripted(
            vlm={0: [("apple", [0, 0, 20, 20]), ("apple", [0, 0, 20, 20])]},
            observe={"apple": "empty_object_pointcloud"},
        )
        result = _run(InspectScene(target="x", exploration_mode=True, pan_angles=[0.0], max_retries=1), ctx)
        assert result.suggested_replan_context["found"] is False
        assert len(scripted.detector_calls) == 1

    def test_all_views_not_found_reports_found_false(self, ctx, scripted):
        scripted(vlm={})
        result = _run(InspectScene(target="x", exploration_mode=True), ctx)
        assert result.status == "needs_replan"
        assert result.force_replan is False
        assert result.suggested_replan_context["found"] is False
        assert _FakeObserve.calls == []
        assert len(scripted.detector_calls) == 3

    def test_verification_can_be_disabled(self, ctx, scripted):
        scripted(vlm={0: [("sandwich", [1, 1, 2, 2])]})
        result = _run(InspectScene(target="x", exploration_mode=True, verify_with_observe=False), ctx)
        assert result.status == "needs_replan"
        assert result.suggested_replan_context["found_label"] == "sandwich"
        assert result.suggested_replan_context["observed"] is False
        assert _FakeObserve.calls == []


class TestDetectorResponseParsing:
    def _detector(self, **kw):
        params = dict(target="something to eat", detector_module="m", pan=0.0, tilt=0.0, index=0)
        params.update(kw)
        return inspect_scene._DetectDescribedTargetInView(**params)

    def test_found_with_confidence(self, ctx):
        d = self._detector()
        d._handle_detection_response(ctx, {"content": '{"visible_objects":[{"label":"apple","box":[1,2,3,4]}],"status":"Found","label":"apple","box":[1,2,3,4],"confidence":0.9}'})
        assert d.check_result(ctx).status == "success"
        assert d.found_label == "apple"
        assert d.found_confidence == 0.9

    def test_low_confidence_rejected(self, ctx):
        d = self._detector(min_confidence=0.6)
        d._handle_detection_response(ctx, {"content": '{"status":"Found","label":"apple","box":[1,2,3,4],"confidence":0.3}'})
        r = d.check_result(ctx)
        assert r.status == "failure" and r.reason.startswith("low_confidence")

    def test_not_found(self, ctx):
        d = self._detector()
        d._handle_detection_response(ctx, {"content": '```json\n{"visible_objects":[],"status":"Not Found","confidence":0.95}\n```'})
        assert d.check_result(ctx).reason == "object_not_found"

    def test_repeated_rejected_label(self, ctx):
        d = self._detector(rejected_labels=["Sandwich"])
        d._handle_detection_response(ctx, {"content": '{"status":"Found","label":"sandwich","box":[1,2,3,4],"confidence":0.8}'})
        assert d.check_result(ctx).reason.startswith("repeated_rejected_label")

    def test_prompt_mentions_rejection(self):
        d = self._detector(rejected_labels=["sandwich"], rejected_box=[1, 2, 3, 4])
        text = d._prompt_text()
        assert "'sandwich'" in text and "[1, 2, 3, 4]" in text and "Not Found" in text


class TestObserveSceneDetectionQuery:
    def test_node_id_resolves_to_label(self, ctx):
        from src.core.scene_graph import SceneGraph
        from src.plugin.actions.observe import ObserveScene

        sg = SceneGraph()
        sg.add_node("apple_36", "object", {"label": "apple"})
        ctx.scene_graph = sg
        assert ObserveScene(target="apple_36")._detection_query(ctx) == "apple"
        assert ObserveScene(target="sandwich")._detection_query(ctx) == "sandwich"

    def test_without_scene_graph_target_is_used(self, ctx):
        from src.plugin.actions.observe import ObserveScene

        ctx.scene_graph = None
        assert ObserveScene(target="apple_36")._detection_query(ctx) == "apple_36"


def test_reentry_refreshes_candidate_and_resets_rejections_without_rebuilding(ctx, scripted):
    import py_trees
    scripted(vlm={0: [("sandwich", [10, 10, 50, 50]), ("banana", [10, 10, 50, 50])]}, observe={"banana": [10, 10, 50, 50]})
    action = InspectScene(target="food", pan_angles=[0.0], exploration_mode=True, max_retries=1)
    first = _run(action, ctx)
    assert first.suggested_replan_context["found_label"] == "banana"
    root = action_node(action, ctx)
    ids = [node.id for node in root.iterate()]
    root.stop(py_trees.common.Status.INVALID)
    ctx.execution.reset_control()
    scripted(vlm={0: [("sandwich", [10, 10, 50, 50])]}, observe={"sandwich": [10, 10, 50, 50]})
    second = _run(action, ctx)
    assert second.suggested_replan_context["found_label"] == "sandwich"
    assert scripted.detector_calls[-1] == {"index": 0, "attempt": 0, "rejected": []}
    assert [node.id for node in root.iterate()] == ids
