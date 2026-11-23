"""Printable end-to-end demonstration of the Boxer SceneGraph API.

Run from the Orchid repository with output capture disabled:

    python3 -m pytest -s tests/integration/test_boxer_scene_graph_demo.py

Override the input paths when needed:

    BOXER_SCENE_GRAPH_PATH=/path/to/scene_graph.json \
    BOXER_RELATIONSHIP_CONFIG_PATH=/path/to/support_relationships.yaml \
    python3 -m pytest -s tests/integration/test_boxer_scene_graph_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.core.scene_graph import (
    Edge,
    Node,
    SceneGraph,
    classify_boxer_label,
    load_relationship_classes,
    quaternion_to_yaw,
)


DEFAULT_SCENE_GRAPH = Path(
    "/home/user/yelchuriv1/boxer/user_output/scene_graph/"
    "scene_graph_access_poses_vlm.json"
)


def _section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def _configured_path(environment_name: str, default: Path | None = None) -> Path | None:
    value = os.getenv(environment_name)
    return Path(value).expanduser() if value else default


def test_every_scene_graph_function() -> None:
    """Load the Boxer graph and print representative output from every API."""
    scene_path = _configured_path("BOXER_SCENE_GRAPH_PATH", DEFAULT_SCENE_GRAPH)
    assert scene_path is not None
    if not scene_path.is_file():
        raise FileNotFoundError(f"Boxer scene graph is not available: {scene_path}")

    raw_graph = json.loads(scene_path.read_text(encoding="utf-8"))
    metadata = raw_graph.get("metadata", {})
    metadata_config = metadata.get("relationship_config")
    relationship_path = _configured_path(
        "BOXER_RELATIONSHIP_CONFIG_PATH",
        Path(metadata_config).expanduser() if metadata_config else None,
    )
    if relationship_path is None or not relationship_path.is_file():
        raise FileNotFoundError(
            f"Relationship config is not available: {relationship_path}"
        )

    _section("load_relationship_classes")
    furniture_classes, null_classes = load_relationship_classes(relationship_path)
    print("YAML:", relationship_path)
    print("Furniture class count:", len(furniture_classes))
    print("Furniture examples:", sorted(furniture_classes)[:8])
    print("Null class count:", len(null_classes))
    print("Null-class examples:", sorted(null_classes)[:8])
    assert "desk" in furniture_classes
    assert "door" in null_classes

    _section("classify_boxer_label")
    for label in ("desk", "door", "cup"):
        result = classify_boxer_label(label, furniture_classes, null_classes)
        print(f"{label!r} -> {result!r}")
    assert classify_boxer_label("desk", furniture_classes, null_classes) == "furniture"
    assert classify_boxer_label("door", furniture_classes, null_classes) == "null_class"
    assert classify_boxer_label("cup", furniture_classes, null_classes) == "object"

    _section("quaternion_to_yaw")
    first_raw_pose = next(
        pose
        for item in raw_graph["objects"]
        for pose in item.get("base_access_poses", [])
    )
    yaw = quaternion_to_yaw(first_raw_pose["orientation"])
    print("Quaternion:", first_raw_pose["orientation"])
    print("Yaw in radians:", yaw)
    assert -3.141593 <= yaw <= 3.141593

    _section("SceneGraph.from_json and __len__")
    graph = SceneGraph.from_json(
        str(scene_path), relationship_config_path=relationship_path
    )
    print("Scene graph:", scene_path)
    print("Total normalized nodes:", len(graph))
    print("Node types:", dict(Counter(node.type for node in graph._nodes.values())))
    print("Edge types:", dict(Counter(edge.relation_type for edge in graph._edges)))
    assert len(graph) > 0

    _section("SceneGraph.from_dict")
    graph_from_dict = SceneGraph.from_dict(raw_graph, relationship_path)
    print("Nodes loaded from dictionary:", len(graph_from_dict))
    assert len(graph_from_dict) == len(graph)

    _section("get_node and Node.to_dict")
    desk = graph.get_node("desk_5")
    print("desk_5 node:", desk)
    assert desk is not None
    print("desk_5 as dictionary:", desk.to_dict())
    assert desk.type == "furniture"

    _section("Node.from_dict")
    reconstructed_node = Node.from_dict(desk.to_dict())
    print("Reconstructed node:", reconstructed_node)
    assert reconstructed_node == desk

    _section("get_node_location")
    desk_location = graph.get_node_location("desk_5")
    room_location = graph.get_node_location("room_01")
    missing_location = graph.get_node_location("does_not_exist")
    print("desk_5 location:", desk_location)
    print("room_01 location:", room_location)
    print("missing-node location:", missing_location)
    assert desk_location is not None and len(desk_location) == 3
    assert room_location is not None and len(room_location) == 3
    assert missing_location is None

    _section("get_node_bbox")
    desk_bbox = graph.get_node_bbox("desk_5")
    room_bbox = graph.get_node_bbox("room_01")
    print("desk_5 axis-aligned bounding box:", desk_bbox)
    print("room_01 bounding box (not supplied by Boxer):", room_bbox)
    assert desk_bbox is not None and len(desk_bbox) == 6
    assert room_bbox is None

    _section("get_node_access_poses")
    access_poses = graph.get_node_access_poses("desk_5")
    print("desk_5 access-pose count:", len(access_poses))
    print("First normalized access pose:", access_poses[0])
    assert access_poses
    assert {"x", "y", "yaw", "frame_id", "rank"} <= access_poses[0].keys()

    _section("get_best_access_pose")
    best_pose = graph.get_best_access_pose("desk_5")
    no_pose = graph.get_best_access_pose("cup_22")
    print("Best desk_5 pose:", best_pose)
    print("Best cup_22 pose:", no_pose)
    assert best_pose is not None
    assert best_pose["rank"] == min(pose["rank"] for pose in access_poses)
    assert no_pose is None

    _section("get_node_visit_locations")
    visit_locations = graph.get_node_visit_locations("desk_5")
    print("Legacy [x, y, yaw] locations:", visit_locations)
    assert visit_locations
    assert all(len(location) == 3 for location in visit_locations)

    _section("get_neighbors")
    room_contents = graph.get_neighbors("room_01", "contains")
    supported_objects = graph.get_neighbors("desk_6", "supports")
    print("First five direct room_01 contents:", [n.id for n in room_contents[:5]])
    print("Objects supported by desk_6:", [n.id for n in supported_objects])
    assert room_contents
    assert supported_objects

    _section("get_edges_from and Edge.to_dict")
    support_edges = graph.get_edges_from("desk_6", "supports")
    print("Support edges from desk_6:", [edge.to_dict() for edge in support_edges])
    assert support_edges

    _section("Edge.from_dict")
    reconstructed_edge = Edge.from_dict(support_edges[0].to_dict())
    print("Reconstructed edge:", reconstructed_edge)
    assert reconstructed_edge == support_edges[0]

    _section("to_dict")
    normalized = graph.to_dict()
    print("Top-level keys:", list(normalized))
    print("Serialized node count:", len(normalized["nodes"]))
    print("Serialized edge count:", len(normalized["edges"]))
    print("Serialized desk_5:", normalized["nodes"]["desk_5"])
    assert len(normalized["nodes"]) == len(graph)

    _section("to_json")
    with tempfile.TemporaryDirectory() as temporary_directory:
        export_path = Path(temporary_directory) / "normalized_scene_graph.json"
        graph.to_json(str(export_path))
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        print("Normalized export path:", export_path)
        print("Export keys:", list(exported))
        print("Export size in bytes:", export_path.stat().st_size)
        assert exported["nodes"]["desk_5"]["type"] == "furniture"
        print("Note: this is a normalized export, not Boxer input for from_json().")

    _section("add_node, add_edge, remove_node, and clear")
    scratch = SceneGraph()
    added_room = scratch.add_node("demo_room", "room", {"name": "Demo Room"})
    added_object = scratch.add_node("demo_object", "object", {"name": "Demo Object"})
    added_edge = scratch.add_edge("demo_room", "demo_object", "contains")
    print("Added room:", added_room)
    print("Added object:", added_object)
    print("Added edge:", added_edge)
    print("Scratch graph length:", len(scratch))
    assert scratch.get_neighbors("demo_room", "contains") == [added_object]

    removed = scratch.remove_node("demo_object")
    print("remove_node returned:", removed)
    print("Edges after removing demo_object:", scratch.get_edges_from("demo_room"))
    assert removed is True
    assert scratch.get_edges_from("demo_room") == []

    scratch.clear()
    print("Scratch graph length after clear:", len(scratch))
    assert len(scratch) == 0

    _section("result")
    print("All SceneGraph API demonstrations and assertions passed.")


if __name__ == "__main__":
    test_every_scene_graph_function()
