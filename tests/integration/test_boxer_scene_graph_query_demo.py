"""Printable demonstration of every Boxer SceneGraphQuery operation.

Run directly:

    python3 tests/integration/test_boxer_scene_graph_query_demo.py

Or, when pytest is installed:

    python3 -m pytest -s tests/integration/test_boxer_scene_graph_query_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.core.scene_graph import SceneGraph
from src.core.scene_graph_query import SceneGraphQuery


DEFAULT_SCENE_GRAPH = Path(
    "/home/user/yelchuriv1/boxer/user_output/scene_graph/"
    "scene_graph_access_poses_vlm.json"
)


def _section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def test_every_scene_graph_query_function() -> None:
    """Print and assert representative results from every query method."""
    scene_path = Path(
        os.getenv("BOXER_SCENE_GRAPH_PATH", str(DEFAULT_SCENE_GRAPH))
    ).expanduser()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Boxer scene graph not found: {scene_path}")

    graph = SceneGraph.from_json(str(scene_path))
    query = SceneGraphQuery(graph)

    _section("get_all_rooms")
    rooms = query.get_all_rooms()
    print(rooms)
    assert {"id": "room_01", "room_type": "office"} in rooms
    assert {"id": "room_03", "room_type": "kitchen"} in rooms

    _section("get_objects_by_type")
    furniture = query.get_objects_by_type("furniture")
    objects = query.get_objects_by_type("object")
    null_classes = query.get_objects_by_type("null_class")
    print("Furniture count and examples:", len(furniture), furniture[:5])
    print("Object count and examples:", len(objects), objects[:5])
    print("Null-class count and examples:", len(null_classes), null_classes[:5])
    assert furniture and objects and null_classes

    _section("get_room_contents")
    room_contents = query.get_room_contents("room_01")
    print("Room:", room_contents["room"])
    print("Furniture count:", len(room_contents["furniture"]))
    print("First furniture entries:", room_contents["furniture"][:3])
    print("Direct objects:", room_contents["objects"])
    print("Direct null classes:", room_contents["null_classes"])
    assert any(
        item["id"] == "desk_6" and "cup_22" in item["objects"]
        for item in room_contents["furniture"]
    )

    _section("support and room relationships")
    supporting_furniture = query.get_supporting_furniture("cup_22")
    compatibility_location = query.get_object_location("cup_22")
    containing_room = query.get_containing_room("cup_22")
    in_room = query.is_object_in_room("cup_22", "room_01")
    print("Supporting furniture:", supporting_furniture)
    print("get_object_location compatibility result:", compatibility_location)
    print("Containing room:", containing_room)
    print("Is cup_22 in room_01?:", in_room)
    assert supporting_furniture == "desk_6"
    assert compatibility_location == "desk_6"
    assert containing_room == "room_01"
    assert in_room is True

    _section("name search and matching")
    desk_matches = query.find_objects_by_name("desk")
    cup_matches = query.find_objects_by_name("cup")
    exact_id_matches = query.find_objects("desk_5")
    missing_matches = query.find_objects("not_in_the_scene")
    ambiguous_navigation = query.get_navigation_target("desk")
    print("All desk matches:", desk_matches)
    print("Cup matches:", cup_matches)
    print("Exact desk_5 matches:", exact_id_matches)
    print("Missing matches:", missing_matches)
    print("Navigation target for ambiguous 'desk':", ambiguous_navigation)
    assert len(desk_matches) > 1
    assert cup_matches == ["cup_22"]
    assert exact_id_matches == ["desk_5"]
    assert missing_matches == []
    assert ambiguous_navigation is None

    _section("geometry")
    location = query.get_object_location_3d("desk_5")
    bbox = query.get_object_bbox("desk_5")
    top_height = query.get_top_height("desk_5")
    info = query.get_object_info("cup_22")
    print("desk_5 center [x, y, z]:", location)
    print("desk_5 axis-aligned bbox:", bbox)
    print("desk_5 top height:", top_height)
    print("cup_22 concise information:", info)
    assert location is not None and len(location) == 3
    assert bbox is not None and len(bbox) == 6
    assert top_height == bbox[5]
    assert info is not None and info["supported_by"] == "desk_6"

    _section("navigation through supporting furniture")
    navigation_target = query.get_navigation_target("cup_22")
    visit_locations = query.get_visit_locations("cup_22")
    best = query.get_best_visit_location("cup_22")
    nearest = query.get_nearest_visit_location("cup_22", [3.0, -2.5, 99.0])
    print("Requested target: cup_22")
    print("Navigation target:", navigation_target)
    print("Access-pose count:", len(visit_locations))
    print("Best ranked pose:", best)
    print("Nearest pose using XY only:", nearest)
    assert navigation_target == "desk_6"
    assert visit_locations
    assert best is not None and best["navigation_target"] == "desk_6"
    assert nearest is not None and nearest["requested_target"] == "cup_22"

    _section("get_scene_description")
    description = query.get_scene_description()
    print(description)
    assert "Room room_01 [office]" in description
    assert "desk_6" in description
    assert "supports cup_22" in description

    _section("result")
    print("All SceneGraphQuery demonstrations and assertions passed.")


if __name__ == "__main__":
    test_every_scene_graph_query_function()
