"""Tests for Nav2 controller dispatch shape."""

from __future__ import annotations

from types import SimpleNamespace

from src.plugin.controllers.nav2 import Nav2Controller


class _DoneFuture:
    def __init__(self, result):
        self._result = result
        self.callback = None

    def result(self):
        return self._result

    def add_done_callback(self, callback):
        self.callback = callback


class _FakeGoalHandle:
    accepted = True

    def __init__(self):
        self.cancel_future = _DoneFuture(SimpleNamespace(goals_canceling=[object()]))

    def get_result_async(self):
        return _DoneFuture(SimpleNamespace(status=4))

    def cancel_goal_async(self):
        return self.cancel_future


def test_nav2_controller_sends_navigate_to_pose_goal(monkeypatch):
    sent = {}

    class FakeGoal:
        def __init__(self):
            self.pose = None

    class FakeNavigateToPose:
        Goal = FakeGoal

    class FakeActionClient:
        def wait_for_server(self, timeout_sec):
            sent["timeout_sec"] = timeout_sec
            return True

        def send_goal_async(self, goal_msg, feedback_callback=None):
            sent["goal_msg"] = goal_msg
            sent["feedback_callback"] = feedback_callback
            return SimpleNamespace(add_done_callback=lambda callback: sent.setdefault("done_callback", callback))

    class FakeClock:
        def now(self):
            from builtin_interfaces.msg import Time

            return SimpleNamespace(to_msg=lambda: Time())

    monkeypatch.setattr(Nav2Controller, "_init_ros", lambda self: None)
    controller = Nav2Controller({"action_name": "navigate_to_pose", "frame_id": "map"})
    controller._NavigateToPose = FakeNavigateToPose
    controller._action_client = FakeActionClient()
    controller._ros_node = SimpleNamespace(get_clock=lambda: FakeClock())

    controller.send_goal((1.0, 2.0, 0.5))

    assert sent["timeout_sec"] == 2.0
    assert sent["goal_msg"].pose.header.frame_id == "map"
    assert sent["goal_msg"].pose.pose.position.x == 1.0
    assert sent["goal_msg"].pose.pose.position.y == 2.0
    assert sent["goal_msg"].pose.pose.orientation.z != 0.0


def test_nav2_result_succeeded_sets_success(monkeypatch):
    monkeypatch.setattr(Nav2Controller, "_init_ros", lambda self: None)
    controller = Nav2Controller({})

    controller._on_result(_DoneFuture(SimpleNamespace(status=4)))

    assert controller.get_status() == "succeeded"
    assert controller.get_error() is None


def test_nav2_result_canceled_sets_idle(monkeypatch):
    monkeypatch.setattr(Nav2Controller, "_init_ros", lambda self: None)
    controller = Nav2Controller({})

    controller._on_result(_DoneFuture(SimpleNamespace(status=5)))

    assert controller.get_status() == "idle"
    assert controller.get_error() is None


def test_nav2_result_aborted_sets_failure(monkeypatch):
    monkeypatch.setattr(Nav2Controller, "_init_ros", lambda self: None)
    controller = Nav2Controller({})

    controller._on_result(_DoneFuture(SimpleNamespace(status=6)))

    assert controller.get_status() == "failed"
    assert "aborted" in (controller.get_error() or "")


def test_nav2_cancel_tracks_cancel_future(monkeypatch):
    monkeypatch.setattr(Nav2Controller, "_init_ros", lambda self: None)
    controller = Nav2Controller({})
    handle = _FakeGoalHandle()
    controller._goal_handle = handle
    controller._status = "in_progress"

    controller.cancel_goal()

    assert controller._cancel_future is handle.cancel_future
    assert handle.cancel_future.callback is not None


def test_nav2_goal_rejection_sets_failure(monkeypatch):
    monkeypatch.setattr(Nav2Controller, "_init_ros", lambda self: None)
    controller = Nav2Controller({})

    controller._on_goal_response(_DoneFuture(SimpleNamespace(accepted=False)))

    assert controller.get_status() == "failed"
    assert controller.get_error() == "Nav2 goal rejected"
