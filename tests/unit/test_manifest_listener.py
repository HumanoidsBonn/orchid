"""Unit tests for ManifestListener implementations."""

import json
from pathlib import Path
import pytest

from src.orchestrator.manifest_listener import FileManifestListener, ManifestListener, Ros2ManifestListener
from src.core.manifest import RobotManifest

def test_manifest_listener_is_abstract():
    """Verify ManifestListener cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ManifestListener()


def test_file_listener_loads_single(tmp_path):
    """Test loading a single manifest from JSON file."""
    src = pytest.manifest_path
    dst = tmp_path / "m1.json"
    print(f"Copying {src} to {dst} for test")
    dst.write_bytes(src.read_bytes())
    listener = FileManifestListener([dst])
    manifests = listener.load_manifests()
    assert isinstance(manifests, list)
    assert len(manifests) == 1
    m = manifests[0]
    assert isinstance(m, RobotManifest)
    assert m.robot_id == "example_bot"


def test_file_listener_loads_multiple(tmp_path):
    """Test loading multiple manifests from multiple JSON files."""
    src = pytest.manifest_path
    dst1 = tmp_path / "m1.json"
    dst1.write_bytes(src.read_bytes())
    
    # Create second manifest with different robot_id
    data = json.loads(src.read_text())
    data["robot_id"] = "example_bot_2"
    dst2 = tmp_path / "m2.json"
    dst2.write_text(json.dumps(data))
    
    listener = FileManifestListener([dst1, dst2])
    manifests = listener.load_manifests()
    assert len(manifests) == 2
    assert manifests[0].robot_id == "example_bot"
    assert manifests[1].robot_id == "example_bot_2"


def test_file_listener_empty_paths_returns_empty():
    """Test FileManifestListener with no paths returns empty list."""
    listener = FileManifestListener([])
    manifests = listener.load_manifests()
    assert manifests == []


def test_file_listener_no_paths_arg_returns_empty():
    """Test FileManifestListener with None paths argument."""
    listener = FileManifestListener()
    manifests = listener.load_manifests()
    assert manifests == []


def test_add_manifest_path_dynamic(tmp_path):
    """Test dynamically adding manifest paths after initialization."""
    listener = FileManifestListener()
    src = pytest.manifest_path
    dst = tmp_path / "m2.json"
    dst.write_bytes(src.read_bytes())
    listener.add_manifest_path(dst)
    manifests = listener.load_manifests()
    assert len(manifests) == 1
    assert manifests[0].robot_id == "example_bot"


def test_load_manifest_file_not_found(tmp_path):
    """Test FileManifestListener raises FileNotFoundError for missing file."""
    p = tmp_path / "nope.json"
    listener = FileManifestListener([p])
    with pytest.raises(FileNotFoundError):
        listener.load_manifests()


def test_file_listener_multiple_add_paths(tmp_path):
    """Test adding multiple paths sequentially."""
    src = pytest.manifest_path
    
    dst1 = tmp_path / "m1.json"
    dst1.write_bytes(src.read_bytes())
    
    data = json.loads(src.read_text())
    data["robot_id"] = "bot_2"
    dst2 = tmp_path / "m2.json"
    dst2.write_text(json.dumps(data))
    
    listener = FileManifestListener()
    listener.add_manifest_path(dst1)
    listener.add_manifest_path(dst2)
    
    manifests = listener.load_manifests()
    assert len(manifests) == 2


@pytest.mark.requires_ros2
def test_ros2_manifest_listener_init():
    """Test Ros2ManifestListener initializes when ROS2 available."""
    rl = Ros2ManifestListener()
    assert hasattr(rl, "load_manifests")
    assert hasattr(rl, "shutdown")
    rl.node.destroy_node()


@pytest.mark.requires_ros2
def test_ros2_manifest_listener_returns_list():
    """Test Ros2ManifestListener load_manifests returns list."""
    rl = Ros2ManifestListener()
    manifests = rl.load_manifests()
    assert isinstance(manifests, list)
    rl.node.destroy_node()
