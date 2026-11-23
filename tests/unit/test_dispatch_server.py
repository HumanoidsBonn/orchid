import pytest
from pathlib import Path
from unittest.mock import Mock

from src.orchestrator.manifest_listener import FileManifestListener
from src.orchestrator.dispatch_server import (
    DEFAULT_MANIFEST_PATH,
    DispatchServer,
    _parse_dispatch_args,
    _resolve_manifest_config,
)

from tests.fixtures.mock_manifests import basic_registry

def _write_example(tmp_path, name="example.json"):
    src = pytest.manifest_path
    dst = tmp_path / name
    dst.write_bytes(src.read_bytes())
    return dst


def test_dispatch_init_without_listener(basic_registry):
    server = DispatchServer(registry=basic_registry)
    assert isinstance(server.manifests, dict)
    assert server.manifests == {}

def test_dispatch_init_with_file_listener(basic_registry, tmp_path):
    p = _write_example(tmp_path)
    listener = FileManifestListener([p])
    server = DispatchServer(registry=basic_registry, manifest_listener=listener)
    assert "example_bot" in server.manifests

def test_listener_load_failure_handled(basic_registry):
    class BadListener:
        def load_manifests(self):
            raise RuntimeError("boom")

    server = DispatchServer(registry=basic_registry, manifest_listener=BadListener())
    assert server.manifests == {}


def test_get_session_creates_robot_session(basic_registry, tmp_path):
    p = _write_example(tmp_path)
    listener = FileManifestListener([p])
    server = DispatchServer(registry=basic_registry, manifest_listener=listener)
    session = server.get_session("example_bot")
    assert session.robot_id == "example_bot"
    assert session.manifest.robot_id == "example_bot"

def test_list_robots_after_listener_load(basic_registry, tmp_path):
    p1 = _write_example(tmp_path, name="m1.json")

    import json
    data = json.loads(pytest.manifest_path.read_text())
    
    data["robot_id"] = "example_bot_2"
    p2 = tmp_path / "m2.json"
    p2.write_text(json.dumps(data))
    listener = FileManifestListener([p1, p2])
    server = DispatchServer(registry=basic_registry, manifest_listener=listener)
    r = server.list_robots()
    assert "example_bot" in r and "example_bot_2" in r


def test_dispatch_cli_manifest_file_overrides_env():
    parsed, ros_args = _parse_dispatch_args(
        [
            "--manifest-file",
            "manifests/a.json",
            "--manifest-file",
            "manifests/b.json",
            "--ros-args",
            "-r",
            "__node:=dispatch",
        ]
    )

    manifest_type, paths = _resolve_manifest_config(
        parsed,
        {
            "CONFIG_DISPATCH_MANIFEST_FILES": "manifests/env.json",
            "CONFIG_DISPATCH_MANIFEST_TYPE": "file",
        },
    )

    assert manifest_type == "file"
    assert [str(path) for path in paths] == ["manifests/a.json", "manifests/b.json"]
    assert ros_args == ["--ros-args", "-r", "__node:=dispatch"]


def test_dispatch_cli_manifest_files_csv():
    parsed, _ = _parse_dispatch_args(["--manifest-files", "one.json, two.json"])
    _, paths = _resolve_manifest_config(parsed, {})

    assert [str(path) for path in paths] == ["one.json", "two.json"]


def test_dispatch_env_manifest_files_used_when_no_cli():
    parsed, _ = _parse_dispatch_args([])
    _, paths = _resolve_manifest_config(
        parsed,
        {"CONFIG_DISPATCH_MANIFEST_FILES": "env_a.json,env_b.json"},
    )

    assert [str(path) for path in paths] == ["env_a.json", "env_b.json"]


def test_dispatch_default_manifest_path():
    parsed, _ = _parse_dispatch_args([])
    manifest_type, paths = _resolve_manifest_config(parsed, {})

    assert manifest_type == "file"
    assert paths == [DEFAULT_MANIFEST_PATH]


@pytest.mark.parametrize("invalid_handles", [1, 3])
def test_dispatch_keeps_spinning_after_destroyed_subscription(monkeypatch, invalid_handles):
    _exercise_executor_failure(monkeypatch, invalid_handles=invalid_handles)


def test_dispatch_unrelated_executor_error_still_propagates(monkeypatch):
    _exercise_executor_failure(monkeypatch, unexpected=True)


def _exercise_executor_failure(monkeypatch, invalid_handles=0, unexpected=False):
    """Exercise main's real loop/cleanup with no ROS nodes, threads or goals."""
    rclpy = pytest.importorskip("rclpy")
    import rclpy.executors
    from rclpy._rclpy_pybind11 import InvalidHandle
    import src.orchestrator.dispatch_server as dispatch

    executor, node, session, server = Mock(), Mock(), Mock(), Mock()
    session.ros2_wrapper.node = node
    server.list_robots.return_value = ["test_robot"]
    server.get_session.return_value = session
    warning = Mock()
    shutdown = Mock()
    monkeypatch.setattr(rclpy, "init", Mock())
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(rclpy, "shutdown", shutdown)
    monkeypatch.setattr(rclpy.executors, "MultiThreadedExecutor", Mock(return_value=executor))
    monkeypatch.setattr(dispatch, "DispatchServer", Mock(return_value=server))
    monkeypatch.setattr(dispatch, "FileManifestListener", Mock())
    monkeypatch.setattr(dispatch, "_resolve_manifest_config", lambda *args: ("file", []))
    monkeypatch.setattr(dispatch.log, "set_speech_mode", Mock())
    monkeypatch.setattr(dispatch.log, "speak", Mock())
    monkeypatch.setattr(dispatch.log, "warn", warning)

    spins = 0

    def spin_once(*, timeout_sec):
        nonlocal spins
        spins += 1
        assert timeout_sec == .1
        # Recovery must not tear down the server/session between iterations.
        executor.shutdown.assert_not_called()
        node.destroy_node.assert_not_called()
        shutdown.assert_not_called()
        if unexpected:
            raise RuntimeError("unrelated executor error")
        if spins <= invalid_handles:
            raise InvalidHandle("cannot use Destroyable because destruction was requested")
        if spins == invalid_handles + 1:
            return  # Successfully process subsequent work after the exception(s).
        raise KeyboardInterrupt()

    executor.spin_once.side_effect = spin_once
    if unexpected:
        with pytest.raises(RuntimeError, match="unrelated executor error"):
            dispatch.main([])
        assert spins == 1 and warning.call_count == 0
    else:
        dispatch.main([])
        assert spins == invalid_handles + 2
        assert warning.call_count == invalid_handles
    session.start.assert_called_once()
    executor.add_node.assert_called_once_with(node)
    executor.shutdown.assert_called_once()
    node.destroy_node.assert_called_once()
    shutdown.assert_called_once()
