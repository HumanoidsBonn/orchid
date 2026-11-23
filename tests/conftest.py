"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest
from src.core.context import Context

# ========== ROS2 Detection & Initialization ==========
try:
    import rclpy  # pragma: no cover - optional in CI
    HAS_RCLPY = True
except Exception:  # pragma: no cover
    HAS_RCLPY = False

# Store in pytest namespace for access in tests
pytest.HAS_RCLPY = HAS_RCLPY

# ========== End ROS2 Detection ==========

from src.core.world_state import WorldState
from src.core.scene_graph import SceneGraph, Node, Edge
from src.core.module_proxy import ModuleProxy
from tests.fixtures.mock_controllers import MockNavController, MockArmController, MockGripperController

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MANIFEST_PATH = REPO_ROOT / "manifests" / "example_manifest.json"


@pytest.fixture
def mock_nav_controller():
    """Fixture: mock navigation controller."""
    return MockNavController({})


@pytest.fixture
def mock_arm_controller():
    """Fixture: mock arm controller."""
    return MockArmController({})


@pytest.fixture
def mock_gripper_controller():
    """Fixture: mock gripper controller."""
    return MockGripperController({})


@pytest.fixture
def world_state():
    """Fixture: fresh WorldState with maxlen=10."""
    return WorldState(maxlen=10)


@pytest.fixture
def scene_graph():
    """Fixture: fresh SceneGraph with sample nodes."""
    sg = SceneGraph()
    
    # Add sample nodes
    node_a = Node(id="A", label="object_A", attributes={"color": "red"})
    node_b = Node(id="B", label="object_B", attributes={"color": "blue"})
    node_c = Node(id="C", label="surface", attributes={"type": "table"})
    
    sg.add_node(node_a)
    sg.add_node(node_b)
    sg.add_node(node_c)
    
    # Add edges
    sg.add_edge(Edge(source="A", target="C", relationship="on", attributes={"distance": 0.5}))
    sg.add_edge(Edge(source="B", target="C", relationship="near", attributes={"distance": 0.8}))
    
    return sg


@pytest.fixture
def module_proxy():
    """Fixture: ModuleProxy for testing module loading."""
    return ModuleProxy()


@pytest.fixture
def context_with_controllers(world_state, mock_nav_controller, mock_arm_controller, mock_gripper_controller):
    """Fixture: Context with all mock controllers."""
    return Context(
        nav=mock_nav_controller,
        arm=mock_arm_controller,
        gripper=mock_gripper_controller,
        world_state=world_state,
    )


@pytest.fixture
def context_minimal():
    """Fixture: Minimal Context with defaults."""
    return Context()


@pytest.fixture(autouse=True)
def reset_imports():
    """Fixture: Reset any state between tests (autouse=True)."""
    yield
    # Add cleanup if needed


@pytest.fixture(scope="session", autouse=True)
def init_rclpy_session():
    """Initialize rclpy once per test session if available.
    
    This fixture:
    1. Checks if ROS2 is available
    2. Calls rclpy.init() once at session start
    3. Cleans up with rclpy.shutdown() at session end
    4. Works correctly if ROS2 is not installed (no-op)
    """

    if not pytest.HAS_RCLPY:
        # ROS2 not available; yield and continue
        pytest.manifest_path = TEST_MANIFEST_PATH
        yield
        return
    
    import rclpy
    
    # Initialize ROS2 once per session
    try:
        rclpy.init()
        print("ROS2 initialized for test session")
        pytest.manifest_path = TEST_MANIFEST_PATH
        
        yield
    finally:
        # Ensure clean shutdown
        try:
            rclpy.shutdown()
        except Exception:  # pragma: no cover
            pass


# Pytest hooks and configuration

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_ros2: mark test as requiring ROS2/rclpy"
    )


def pytest_runtest_setup(item):
    """Skip tests marked requires_ros2 if ROS2 not available."""
    if "requires_ros2" in item.keywords and not pytest.HAS_RCLPY:
        pytest.skip("ROS2 not available")


@pytest.fixture(scope="session")
def test_session_id():
    """Session-wide fixture for test IDs."""
    import uuid
    return str(uuid.uuid4())[:8]
