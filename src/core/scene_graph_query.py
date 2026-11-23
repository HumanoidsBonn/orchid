"""Deterministic queries over Orchid's normalized Boxer scene graph."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import src.core.logger as log

if TYPE_CHECKING:
    from src.core.scene_graph import SceneGraph


class SceneGraphQueryHelper:
    """High-level Boxer scene queries used by planning and robot actions."""

    def __init__(self, scene_graph: SceneGraph | None):
        """Initialize the query helper.

        Args:
            scene_graph: Scene graph to query, or None when no graph is loaded.
        """
        self.scene_graph = scene_graph

    def get_all_rooms(self) -> list[dict[str, str]]:
        """Get all rooms with their semantic room types.

        Returns:
            List of dictionaries containing id and room_type, sorted by room
            ID. The list is empty when no scene graph is available.
        """
        if not self.scene_graph:
            return []

        rooms = [
            {
                "id": node.id,
                "room_type": str(node.attributes.get("room_type") or "unknown"),
            }
            for node in self.scene_graph.get_nodes("room")
        ]
        return sorted(rooms, key=lambda room: room["id"])

    def get_objects_by_type(self, object_type: str) -> list[str]:
        """Get all node IDs having a semantic type.

        Args:
            object_type: Node type such as room, furniture, object, or
                null_class.

        Returns:
            Sorted list of matching node IDs.
        """
        if not self.scene_graph:
            return []
        return sorted(
            node.id for node in self.scene_graph.get_nodes(object_type)
        )

    def get_room_contents(self, room_id: str) -> dict[str, Any]:
        """Get furniture and supported objects belonging to a room.

        Args:
            room_id: Exact room node ID.

        Returns:
            Dictionary containing the room ID, furniture entries with their
            supported object IDs, directly contained object IDs, and directly
            contained null-class IDs. Empty collections are returned when the
            graph or room is unavailable.
        """
        empty = {
            "room": room_id,
            "furniture": [],
            "objects": [],
            "null_classes": [],
        }
        if not self.scene_graph:
            return empty

        room = self.scene_graph.get_node(room_id)
        if room is None or room.type != "room":
            return empty

        result = {
            "room": room_id,
            "furniture": [],
            "objects": [],
            "null_classes": [],
        }
        direct_contents = self.scene_graph.get_neighbors(room_id, "contains")
        for node in sorted(direct_contents, key=lambda item: item.id):
            if node.type == "furniture":
                supported = self.scene_graph.get_neighbors(node.id, "supports")
                result["furniture"].append(
                    {
                        "id": node.id,
                        "objects": sorted(item.id for item in supported),
                    }
                )
            elif node.type == "null_class":
                result["null_classes"].append(node.id)
            else:
                result["objects"].append(node.id)

        return result

    def get_supporting_furniture(self, object_id: str) -> str | None:
        """Get the furniture supporting an object.

        Args:
            object_id: Exact object node ID.

        Returns:
            Supporting furniture node ID, or None when the object is missing
            or has no furniture support relationship.
        """
        if not self.scene_graph or not self.scene_graph.get_node(object_id):
            return None

        for edge in self.scene_graph.get_edges_to(object_id, "supports"):
            support = self.scene_graph.get_node(edge.src_id)
            if support is not None and support.type == "furniture":
                return support.id
        return None

    def get_object_location(self, object_id: str) -> str | None:
        """Get the furniture supporting an object.

        This method preserves the previous public name. New code should prefer
        get_supporting_furniture() because this result is not a 3D location.

        Args:
            object_id: Exact object node ID.

        Returns:
            Supporting furniture node ID, or None when unavailable.
        """
        return self.get_supporting_furniture(object_id)

    def get_containing_room(self, node_id: str) -> str | None:
        """Get the room containing an object or furniture node.

        Support relationships are followed from an object to its furniture,
        followed by the furniture-to-room containment relationship.

        Args:
            node_id: Exact object, furniture, or room node ID.

        Returns:
            Containing room node ID, or None when no room can be determined.
        """
        if not self.scene_graph or not self.scene_graph.get_node(node_id):
            return None

        current_id = node_id
        visited: set[str] = set()
        for _ in range(10):
            if current_id in visited:
                return None
            visited.add(current_id)

            current = self.scene_graph.get_node(current_id)
            if current is not None and current.type == "room":
                return current.id

            support_edges = self.scene_graph.get_edges_to(
                current_id, "supports"
            )
            if support_edges:
                current_id = support_edges[0].src_id
                continue

            containment_edges = self.scene_graph.get_edges_to(
                current_id, "contains"
            )
            if containment_edges:
                current_id = containment_edges[0].src_id
                continue

            return None

        return None

    def is_object_in_room(self, object_id: str, room_id: str) -> bool:
        """Check whether an object or furniture belongs to a room.

        Args:
            object_id: Exact object or furniture node ID.
            room_id: Exact room node ID.

        Returns:
            True when the node is directly or indirectly in the room.
        """
        return self.get_containing_room(object_id) == room_id

    @staticmethod
    def _normalise_name(value: str) -> str:
        """Normalize a node name for exact comparison.

        Args:
            value: User-provided or stored node name.

        Returns:
            Lowercase underscore-separated name.
        """
        return "_".join(value.strip().lower().replace("-", " ").split())

    def find_objects(self, name_or_id: str) -> list[str]:
        """Find every object or furniture node matching a name or ID.

        Exact node IDs always return a one-item list. Labels and display names
        can return multiple IDs; callers decide how to select among them.

        Args:
            name_or_id: Exact node ID, Boxer label, or display name.

        Returns:
            Sorted list of matching node IDs. The list is empty when no match
            exists.
        """
        if not self.scene_graph or not isinstance(name_or_id, str):
            return []

        exact = self.scene_graph.get_node(name_or_id)
        if exact is not None:
            return [exact.id]

        query = self._normalise_name(name_or_id)
        matches: list[str] = []
        for node in self.scene_graph.get_nodes():
            if node.type not in {"object", "furniture"}:
                continue
            label = self._normalise_name(str(node.attributes.get("label", "")))
            display_name = self._normalise_name(
                str(node.attributes.get("name", ""))
            )
            if query in {label, display_name}:
                matches.append(node.id)
        if matches:
            return sorted(matches)

        # LLMs frequently trim or reword a node id ("lowtable_19" for
        # "kitchen lowtable_19"). The numeric suffix is unique per graph, so
        # fall back to it when the query carries one and exactly one node
        # shares it.
        suffix = re.search(r"_(\d+)$", name_or_id.strip())
        if suffix:
            wanted = f"_{suffix.group(1)}"
            by_suffix = [
                node.id
                for node in self.scene_graph.get_nodes()
                if node.type in {"object", "furniture"} and node.id.endswith(wanted)
            ]
            if len(by_suffix) == 1:
                log.info(
                    "Resolved '%s' to '%s' by numeric id suffix", name_or_id, by_suffix[0]
                )
                return by_suffix
        return sorted(matches)

    def find_objects_by_name(self, name: str) -> list[str]:
        """Find all objects and furniture matching a label or display name.

        Args:
            name: Boxer label or display name.

        Returns:
            Sorted list of matching node IDs.
        """
        return self.find_objects(name)

    def _get_unique_object_id(self, name_or_id: str) -> str | None:
        """Resolve a name or ID only when it has exactly one match.

        Args:
            name_or_id: Exact node ID, Boxer label, or display name.

        Returns:
            Unique matching node ID, or None for zero or multiple matches.
        """
        matches = self.find_objects(name_or_id)
        return matches[0] if len(matches) == 1 else None

    def get_object_location_3d(self, object_id: str) -> list[float] | None:
        """Get the center location of an object or furniture item.

        Args:
            object_id: Exact node ID.

        Returns:
            Center position [x, y, z], or None when unavailable.
        """
        if not self.scene_graph:
            return None
        return self.scene_graph.get_node_location(object_id)

    def get_object_bbox(self, object_id: str) -> list[float] | None:
        """Get the axis-aligned bounding box of a scene node.

        Args:
            object_id: Exact node ID.

        Returns:
            Bounding box [xmin, ymin, zmin, xmax, ymax, zmax], or None when
            unavailable.
        """
        if not self.scene_graph:
            return None
        return self.scene_graph.get_node_bbox(object_id)

    def get_top_height(self, object_id: str) -> float | None:
        """Get the top height of an object or furniture bounding box.

        Args:
            object_id: Exact node ID.

        Returns:
            Maximum bounding-box z value, or None when unavailable.
        """
        bbox = self.get_object_bbox(object_id)
        return float(bbox[5]) if bbox else None

    def get_object_info(self, object_id: str) -> dict[str, Any] | None:
        """Get concise Boxer information for one exact node ID.

        Args:
            object_id: Exact node ID.

        Returns:
            Dictionary containing identity, geometry, room, support, and access
            pose information, or None when the node does not exist.
        """
        if not self.scene_graph:
            return None

        node = self.scene_graph.get_node(object_id)
        if node is None:
            return None

        return {
            "id": node.id,
            "type": node.type,
            "label": node.attributes.get("label"),
            "name": node.attributes.get("name", node.id),
            "location_3d": self.get_object_location_3d(object_id),
            "bbox": self.get_object_bbox(object_id),
            "top_height": self.get_top_height(object_id),
            "room_id": self.get_containing_room(object_id),
            "supported_by": self.get_supporting_furniture(object_id),
            "access_pose_count": len(
                self.scene_graph.get_node_access_poses(object_id)
            ),
        }

    def get_navigation_target(self, name_or_id: str) -> str | None:
        """Get a uniquely identified node having navigation access poses.

        When the requested object has no poses, its supporting furniture is
        used. Ambiguous names return None instead of selecting arbitrarily.

        Args:
            name_or_id: Exact node ID or a uniquely matching label/name.

        Returns:
            Node ID owning the access poses, or None when the target is
            missing, ambiguous, or inaccessible.
        """
        if not self.scene_graph:
            return None

        node_id = self._get_unique_object_id(name_or_id)
        if node_id is None:
            return None
        if self.scene_graph.get_node_access_poses(node_id):
            return node_id

        supporting_furniture = self.get_supporting_furniture(node_id)
        if (
            supporting_furniture is not None
            and self.scene_graph.get_node_access_poses(supporting_furniture)
        ):
            return supporting_furniture
        return None

    def get_visit_locations(self, name_or_id: str) -> list[dict[str, Any]]:
        """Get normalized access poses for an object or furniture target.

        Object targets inherit the access poses of their supporting furniture.

        Args:
            name_or_id: Exact node ID or a uniquely matching label/name.

        Returns:
            List of normalized access-pose dictionaries. The list is empty
            when the target is missing, ambiguous, or inaccessible.
        """
        if not self.scene_graph:
            return []

        navigation_target = self.get_navigation_target(name_or_id)
        if navigation_target is None:
            return []
        return self.scene_graph.get_node_access_poses(navigation_target)

    @staticmethod
    def _annotate_pose(
        pose: dict[str, Any],
        requested_target: str,
        navigation_target: str,
    ) -> dict[str, Any]:
        """Add target provenance to a selected access pose.

        Args:
            pose: Normalized access-pose dictionary.
            requested_target: Exact node requested by the caller.
            navigation_target: Node whose access pose is being used.

        Returns:
            Copy of the pose containing both target IDs.
        """
        result = dict(pose)
        result["requested_target"] = requested_target
        result["navigation_target"] = navigation_target
        return result

    def get_best_visit_location(
        self, name_or_id: str
    ) -> dict[str, Any] | None:
        """Get the highest-ranked valid access pose for a target.

        Args:
            name_or_id: Exact node ID or a uniquely matching label/name.

        Returns:
            Annotated access-pose dictionary, or None when unavailable.
        """
        if not self.scene_graph:
            return None

        requested_target = self._get_unique_object_id(name_or_id)
        if requested_target is None:
            return None
        navigation_target = self.get_navigation_target(requested_target)
        if navigation_target is None:
            return None
        pose = self.scene_graph.get_best_access_pose(navigation_target)
        if pose is None:
            return None
        return self._annotate_pose(
            pose, requested_target, navigation_target
        )

    def get_nearest_visit_location(
        self,
        name_or_id: str,
        current_position: list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, Any] | None:
        """Get the valid access pose nearest to the robot in XY.

        The returned pose still contains z and yaw, but neither value is used
        in the planar distance calculation. Without a current position, the
        highest-ranked Boxer pose is returned.

        Args:
            name_or_id: Exact node ID or a uniquely matching label/name.
            current_position: Robot position containing at least [x, y].

        Returns:
            Annotated nearest access-pose dictionary, or None when unavailable.

        Raises:
            ValueError: If current_position contains fewer than two values.
        """
        if current_position is None:
            return self.get_best_visit_location(name_or_id)
        if len(current_position) < 2:
            raise ValueError("Current position requires at least [x, y]")

        requested_target = self._get_unique_object_id(name_or_id)
        if requested_target is None:
            return None
        navigation_target = self.get_navigation_target(requested_target)
        if navigation_target is None:
            return None

        poses = [
            pose
            for pose in self.get_visit_locations(requested_target)
            if pose.get("valid", True)
        ]
        if not poses:
            return None

        current_x = float(current_position[0])
        current_y = float(current_position[1])
        nearest = min(
            poses,
            key=lambda pose: math.hypot(
                float(pose["x"]) - current_x,
                float(pose["y"]) - current_y,
            ),
        )
        return self._annotate_pose(
            nearest, requested_target, navigation_target
        )

    @staticmethod
    def _format_location(location: list[float] | None) -> str:
        """Format a location for the planner's scene description.

        Args:
            location: Position [x, y, z], or None.

        Returns:
            Compact formatted location string.
        """
        if not location:
            return ""
        rounded = ", ".join(f"{value:.3f}" for value in location)
        return f" at [{rounded}]"

    def get_scene_description(self) -> str:
        """Get a concise hierarchical description for the LLM planner.

        Returns:
            Text containing rooms, furniture, supported objects, locations,
            access-pose counts, and unassigned null-class nodes.
        """
        if not self.scene_graph:
            return "No scene graph available."

        rooms = self.get_all_rooms()
        lines = [f"Scene contains {len(rooms)} rooms."]

        for room_info in rooms:
            room_id = room_info["id"]
            room_type = room_info["room_type"]
            room_location = self.get_object_location_3d(room_id)
            lines.append(
                f"\nRoom {room_id} [{room_type}]"
                f"{self._format_location(room_location)}:"
            )

            contents = self.get_room_contents(room_id)
            for furniture in contents["furniture"]:
                furniture_id = furniture["id"]
                node = self.scene_graph.get_node(furniture_id)
                if node is None:
                    continue
                pose_count = len(
                    self.scene_graph.get_node_access_poses(furniture_id)
                )
                location = self.get_object_location_3d(furniture_id)
                reach = (
                    f"{pose_count} access poses"
                    if pose_count
                    else "UNREACHABLE: 0 access poses, MoveToTarget cannot go here"
                )
                lines.append(
                    f"- {furniture_id} [furniture, {reach}]"
                    f"{self._format_location(location)}"
                )
                for object_id in furniture["objects"]:
                    object_node = self.scene_graph.get_node(object_id)
                    label = (
                        object_node.attributes.get("label")
                        if object_node is not None
                        else None
                    )
                    label_text = f", label={label}" if label else ""
                    lines.append(
                        f"  - supports {object_id} [object{label_text}]"
                    )

            for object_id in contents["objects"]:
                node = self.scene_graph.get_node(object_id)
                node_type = node.type if node is not None else "object"
                location = self.get_object_location_3d(object_id)
                lines.append(
                    f"- {object_id} [{node_type}]"
                    f"{self._format_location(location)}"
                )

            for node_id in contents["null_classes"]:
                lines.append(f"- {node_id} [null_class]")

        unassigned = [
            node.id
            for node in self.scene_graph.get_nodes()
            if node.type == "null_class"
            and self.get_containing_room(node.id) is None
        ]
        if unassigned:
            lines.append("\nUnassigned null-class nodes:")
            lines.extend(f"- {node_id}" for node_id in sorted(unassigned))

        return "\n".join(lines)


# Shorter name for new code; retain the existing name used by LLMPlanner.
SceneGraphQuery = SceneGraphQueryHelper
