"""Unit tests for WorldState module."""

import json

import numpy as np
import pytest
from src.core.controllers import Pose3D
from src.core.serialization import to_json_safe
from src.core.world_state import ActionRecord, WorldState
from src.core.action import ActionResult


class TestActionRecord:
    """Tests for ActionRecord dataclass."""
    
    def test_action_record_creation(self):
        """Test creating an ActionRecord."""
        result = ActionResult(status="success")
        record = ActionRecord(
            action_name="TestAction",
            params={"param1": "value1"},
            result=result,
            timestamp=1000.0,
            duration_s=0.5
        )
        assert record.action_name == "TestAction"
        assert record.params == {"param1": "value1"}
        assert record.duration_s == 0.5
    
    def test_action_record_to_dict(self):
        """Test ActionRecord serialization."""
        result = ActionResult(status="success")
        record = ActionRecord(
            action_name="TestAction",
            params={"x": 1},
            result=result,
            timestamp=1000.0,
            duration_s=1.5
        )
        
        data = record.to_dict()
        assert data["action_name"] == "TestAction"
        assert data["duration_s"] == 1.5

    def test_action_record_to_dict_sanitizes_pose_and_numpy_payloads(self):
        result = ActionResult(
            status="success",
            payload={"matrix": np.eye(2), "score": np.float32(0.5)},
            suggested_replan_context={"pose": Pose3D(x=1.0, y=2.0, z=3.0)},
        )
        record = ActionRecord(
            action_name="MoveArm",
            params={"pose": Pose3D(x=0.1, y=0.2, z=0.3, frame_id="odom")},
            result=result,
            timestamp=1000.0,
            duration_s=1.5,
        )

        data = record.to_dict()

        json.dumps(data)
        assert data["params"]["pose"]["frame_id"] == "odom"
        assert data["result"]["payload"]["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
        assert data["result"]["suggested_replan_context"]["pose"]["z"] == 3.0


def test_to_json_safe_handles_runtime_payloads():
    class FakeRosMsg:
        def __init__(self):
            self.x = np.float32(0.5)
            self.child = Pose3D(x=1.0, y=2.0, z=3.0)

        def get_fields_and_field_types(self):
            return {"x": "float32", "child": "Pose3D"}

    value = {
        "pose": Pose3D(x=1.0, y=2.0, z=3.0),
        "array": np.array([1, 2, 3]),
        "scalar": np.float32(1.25),
        "items": ({1, 2},),
        "ros_msg": FakeRosMsg(),
    }

    safe = to_json_safe(value)

    json.dumps(safe)
    assert safe["pose"]["x"] == 1.0
    assert safe["array"] == [1, 2, 3]
    assert safe["scalar"] == pytest.approx(1.25)
    assert safe["ros_msg"]["x"] == pytest.approx(0.5)
    assert safe["ros_msg"]["child"]["z"] == 3.0


class TestWorldState:
    """Tests for WorldState."""
    
    def test_worldstate_creation(self):
        """Test WorldState initialization."""
        ws = WorldState(max_history=10)
        assert ws.get("nonexistent") is None
    
    def test_set_and_get_observation(self):
        """Test storing and retrieving observations."""
        ws = WorldState()
        ws.set("gripper_state", {"open": True})
        assert ws.get("gripper_state") == {"open": True}
    
    def test_update_observation(self):
        """Test updating observations."""
        ws = WorldState()
        ws.set("position", {"x": 0.0})
        ws.set("position", {"x": 1.0, "y": 2.0})
        
        obs = ws.get("position")
        assert obs == {"x": 1.0, "y": 2.0}
    
    def test_append_action_record(self):
        """Test appending action records."""
        ws = WorldState(max_history=5)
        result = ActionResult(status="success")
        record = ActionRecord(
            action_name="A1",
            params={},
            result=result,
            timestamp=1000.0,
            duration_s=0.1
        )
        ws.append_action_record(record)
        
        history = ws.get_action_history()
        assert len(history) == 1
        assert history[0].action_name == "A1"
    
    def test_action_history_eviction(self):
        """Test automatic eviction when maxlen exceeded."""
        ws = WorldState(max_history=3)
        result = ActionResult(status="success")
        
        # Add 5 records (overflow by 2)
        for i in range(5):
            record = ActionRecord(
                action_name=f"A{i}",
                params={},
                result=result,
                timestamp=1000.0 + i,
                duration_s=0.1
            )
            ws.append_action_record(record)
        
        # Should only keep last 3
        history = ws.get_action_history()
        assert len(history) == 3
        assert history[0].action_name == "A2"
        assert history[-1].action_name == "A4"
    
    def test_batch_update(self):
        """Test batch update of observations."""
        ws = WorldState()
        ws.update({
            "obs1": {"val": 1},
            "obs2": {"val": 2}
        })
        
        assert ws.get("obs1") == {"val": 1}
        assert ws.get("obs2") == {"val": 2}
    
    def test_to_dict_serialization(self):
        """Test WorldState serialization to dict."""
        ws = WorldState()
        ws.set("test_obs", {"data": 42})
        
        result = ActionResult(status="success")
        record = ActionRecord(
            action_name="TestAction",
            params={},
            result=result,
            timestamp=1000.0,
            duration_s=0.1
        )
        ws.append_action_record(record)
        
        data = ws.to_dict()
        assert "observations" in data
        assert "action_history" in data
        assert "scene_graph" in data
        assert data["observations"]["test_obs"]["data"] == 42
        assert len(data["action_history"]) == 1

    def test_movable_object_state_freshness_and_serialization(self):
        ws = WorldState()
        pose = np.eye(4, dtype=np.float32)

        state = ws.set_movable_object_state(
            "mug",
            pose_matrix=pose,
            frame_id="odom",
            source_action="ObserveScene",
            object_pcd_key="mug_object_pcd",
            surround_pcd_key="mug_surround_pcd",
            source_camera_frame="head_rgbd_sensor_link",
            timestamp=100.0,
        )

        assert state.object_id == "mug"
        assert ws.get_movable_object_state("mug", ttl_s=10.0, now=105.0) is state
        assert ws.get_movable_object_state("mug", ttl_s=1.0, now=105.0) is None
        data = ws.to_dict()
        json.dumps(data)
        assert data["observations"]["mug_movable_object_state"]["frame_id"] == "odom"

    def test_movable_object_reach_target_uses_object_points_top(self):
        ws = WorldState()
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [1.0, 2.0, 0.5]
        ws.set("mug_object_pcd", np.array([[0.9, 2.0, 0.55], [1.1, 2.0, 0.70]], dtype=np.float32))
        ws.set_movable_object_state("mug", pose_matrix=pose, frame_id="odom", object_pcd_key="mug_object_pcd")

        target = ws.get_movable_object_reach_target("mug", clearance_m=0.05)

        assert target is not None
        assert target.x == pytest.approx(1.0)
        assert target.y == pytest.approx(2.0)
        assert target.z == pytest.approx(0.75)
        assert target.frame_id == "odom"
        assert target.source == "object_top_plus_margin"

    def test_movable_object_reach_target_falls_back_to_centroid_height_guess(self):
        ws = WorldState()
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [1.0, 2.0, 0.5]
        ws.set_movable_object_state("mug", pose_matrix=pose, frame_id="odom")

        target = ws.get_movable_object_reach_target("mug", clearance_m=0.05, height_guess_m=0.20)

        assert target is not None
        assert target.z == pytest.approx(0.65)
        assert target.source == "centroid_plus_height_guess"

    def test_movable_object_pick_z_increment_uses_max_of_request_and_height(self):
        ws = WorldState()
        ws.set("mug_object_pcd", np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.42]], dtype=np.float32))
        ws.set_movable_object_state("mug", pose_matrix=np.eye(4), frame_id="odom", object_pcd_key="mug_object_pcd")

        z_inc, metadata = ws.get_movable_object_pick_z_increment(
            "mug",
            requested_z_inc_m=0.1,
            clearance_m=0.05,
        )

        assert z_inc == pytest.approx(0.27)
        assert metadata["source"] == "object_height_plus_clearance"
        assert metadata["object_height_m"] == pytest.approx(0.22)

    def test_movable_object_pick_z_increment_keeps_larger_requested_offset(self):
        ws = WorldState()
        ws.set("mug_object_pcd", np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.25]], dtype=np.float32))

        z_inc, metadata = ws.get_movable_object_pick_z_increment(
            "mug",
            requested_z_inc_m=0.3,
            clearance_m=0.05,
        )

        assert z_inc == pytest.approx(0.3)
        assert metadata["source"] == "requested"
