"""In-memory scene graph built from Boxer scene-graph output.

Boxer stores detected objects in an objects list. Orchid normalizes that list
into nodes and edges so consumers do not depend on the Boxer JSON schema.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


def load_relationship_classes(
    config_path: str | Path,
) -> tuple[set[str], set[str]]:
    """Load furniture and null classes from Boxer's relationship YAML.

    Args:
        config_path: Path to support_relationships.yaml.

    Returns:
        Tuple containing the static-furniture labels and null-class labels.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML structure or class lists are invalid.
    """
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Support relationship config not found: {path}")

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Support relationship config must be a mapping: {path}")

    furniture = config.get("static_furniture")
    null_classes = config.get("null_classes")
    if not isinstance(furniture, list) or not all(
        isinstance(label, str) for label in furniture
    ):
        raise ValueError(f"'static_furniture' must be a list of strings: {path}")
    if not isinstance(null_classes, list) or not all(
        isinstance(label, str) for label in null_classes
    ):
        raise ValueError(f"'null_classes' must be a list of strings: {path}")

    return set(furniture), set(null_classes)


def classify_boxer_label(
    label: str,
    static_furniture: set[str],
    null_classes: set[str],
) -> str:
    """Classify a Boxer label using the relationship configuration.

    Args:
        label: Boxer object-class label.
        static_furniture: Labels classified as furniture.
        null_classes: Labels classified as null classes.

    Returns:
        One of furniture, null_class, or object.
    """
    if label in static_furniture:
        return "furniture"
    if label in null_classes:
        return "null_class"
    return "object"


def quaternion_to_yaw(orientation: dict[str, Any]) -> float:
    """Convert a quaternion mapping to planar yaw.

    Args:
        orientation: Quaternion with x, y, z, and w components.

    Returns:
        Yaw angle in radians.
    """
    x = float(orientation.get("x", 0.0))
    y = float(orientation.get("y", 0.0))
    z = float(orientation.get("z", 0.0))
    w = float(orientation.get("w", 1.0))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


@dataclass
class Node:
    """A node in the scene graph."""

    id: str
    type: str  # e.g., "object", "location", "robot"
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the node to a dictionary.

        Returns:
            Dictionary containing the node ID, type, and attributes.
        """
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> Node:
        """Create a node from a dictionary.

        Args:
            data: Dictionary containing node data.

        Returns:
            Reconstructed node.
        """
        return Node(
            id=data["id"],
            type=data["type"],
            attributes=data.get("attributes", {}),
        )


@dataclass
class Edge:
    """An edge in the scene graph."""

    src_id: str
    dst_id: str
    relation_type: str  # e.g., "on", "near", "holding", "contains"

    def to_dict(self) -> dict:
        """Serialize the edge to a dictionary.

        Returns:
            Dictionary containing source, destination, and relation type.
        """
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> Edge:
        """Create an edge from a dictionary.

        Args:
            data: Dictionary containing edge data.

        Returns:
            Reconstructed edge.
        """
        return Edge(
            src_id=data["src_id"],
            dst_id=data["dst_id"],
            relation_type=data["relation_type"],
        )


class SceneGraph:
    """Spatial graph of scene objects and their relationships."""

    def __init__(self):
        """Initialize empty scene graph."""
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self.revision = 0

    def add_node(
        self, node_id: str, node_type: str, attributes: dict[str, Any] | None = None
    ) -> Node:
        """Add or update a node.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (e.g., "object", "location")
            attributes: Optional dict of attributes (e.g., {"color": "red"})

        Returns:
            The created/updated Node
        """
        node = Node(id=node_id, type=node_type, attributes=attributes or {})
        self._nodes[node_id] = node
        self.revision += 1
        return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges.

        Args:
            node_id: ID of node to remove

        Returns:
            True if node was removed, False if not found
        """
        if node_id not in self._nodes:
            return False

        del self._nodes[node_id]
        self.revision += 1
        # Remove edges involving this node
        self._edges = [
            e for e in self._edges if e.src_id != node_id and e.dst_id != node_id
        ]
        return True

    def get_node(self, node_id: str) -> Node | None:
        """Retrieve a node by ID.

        Args:
            node_id: ID of node to retrieve

        Returns:
            The Node, or None if not found
        """
        return self._nodes.get(node_id)

    def get_node_location(self, node_id: str) -> list[float] | None:
        """Get the 3D location (x, y, z) of a node.

        Args:
            node_id: ID of the node

        Returns:
            List [x, y, z] or None if not found/not set
        """
        node = self.get_node(node_id)
        if not node:
            return None
        loc = node.attributes.get("location")
        if isinstance(loc, (list, tuple)) and len(loc) >= 3:
            return list(loc[:3])
        return None

    def get_node_bbox(self, node_id: str) -> list[float] | None:
        """Get the 3D bounding box of a node.

        Returns [x_min, y_min, z_min, x_max, y_max, z_max] in z-up coordinates.

        Args:
            node_id: ID of the node

        Returns:
            List [x_min, y_min, z_min, x_max, y_max, z_max] or None if not found
        """
        node = self.get_node(node_id)
        if not node:
            return None
        bbox = node.attributes.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 6:
            return list(bbox[:6])

        center = node.attributes.get("center")
        size = node.attributes.get("size")
        if not (
            isinstance(center, (list, tuple))
            and isinstance(size, (list, tuple))
            and len(center) >= 3
            and len(size) >= 3
        ):
            return None

        half_size = [float(size[index]) / 2.0 for index in range(3)]
        rotation = node.attributes.get("rotation")
        if (
            isinstance(rotation, list)
            and len(rotation) >= 3
            and all(isinstance(row, list) and len(row) >= 3 for row in rotation[:3])
        ):
            half_extent = [
                sum(abs(float(rotation[row][column])) * half_size[column]
                    for column in range(3))
                for row in range(3)
            ]
        else:
            half_extent = half_size

        center_xyz = [float(center[index]) for index in range(3)]
        return [
            center_xyz[0] - half_extent[0],
            center_xyz[1] - half_extent[1],
            center_xyz[2] - half_extent[2],
            center_xyz[0] + half_extent[0],
            center_xyz[1] + half_extent[1],
            center_xyz[2] + half_extent[2],
        ]

    def get_node_visit_locations(self, node_id: str) -> list[list[float]]:
        """Get legacy navigation goals for a node.

        Args:
            node_id: Exact node ID.

        Returns:
            List of navigation goals represented as [x, y, yaw].
        """
        node = self.get_node(node_id)
        if not node:
            return []
        visits = node.attributes.get("visit_locations")
        if isinstance(visits, list):
            return [list(v) if isinstance(v, (list, tuple)) else v for v in visits]
        return []

    def get_node_access_poses(self, node_id: str) -> list[dict[str, Any]]:
        """Get normalized Boxer access poses for a node.

        Args:
            node_id: Exact node ID.

        Returns:
            List of normalized access-pose dictionaries.
        """
        node = self.get_node(node_id)
        if not node:
            return []
        poses = node.attributes.get("access_poses")
        if not isinstance(poses, list):
            return []
        return [dict(pose) for pose in poses if isinstance(pose, dict)]

    def get_best_access_pose(self, node_id: str) -> dict[str, Any] | None:
        """Get the highest-ranked valid access pose for a node.

        Args:
            node_id: Exact node ID.

        Returns:
            Highest-ranked valid access-pose dictionary, or None when no valid
            pose is available.
        """
        valid_poses = [
            pose
            for pose in self.get_node_access_poses(node_id)
            if pose.get("valid", True)
        ]
        if not valid_poses:
            return None
        return min(valid_poses, key=lambda pose: pose.get("rank", float("inf")))

    def add_edge(self, src_id: str, dst_id: str, relation_type: str) -> Edge:
        """Add an edge between two nodes.

        Args:
            src_id: Source node ID
            dst_id: Destination node ID
            relation_type: Type of relationship (e.g., "on", "near")

        Returns:
            The created Edge
        """
        edge = Edge(src_id=src_id, dst_id=dst_id, relation_type=relation_type)
        self._edges.append(edge)
        self.revision += 1
        return edge

    def get_neighbors(
        self, node_id: str, relation_type: str | None = None
    ) -> list[Node]:
        """Get all nodes reachable from a given node.

        Args:
            node_id: Source node ID
            relation_type: Optional filter by edge type

        Returns:
            List of neighbor Nodes
        """
        neighbors = []
        for edge in self._edges:
            if edge.src_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbor = self._nodes.get(edge.dst_id)
                    if neighbor:
                        neighbors.append(neighbor)
        return neighbors

    def get_nodes(self, node_type: str | None = None) -> list[Node]:
        """Get all nodes, optionally filtered by semantic type.

        Args:
            node_type: Optional type such as room, furniture, object, or
                null_class.

        Returns:
            List of matching nodes.
        """
        nodes = list(self._nodes.values())
        if node_type is not None:
            nodes = [node for node in nodes if node.type == node_type]
        return nodes

    def get_edges_to(
        self, node_id: str, relation_type: str | None = None
    ) -> list[Edge]:
        """Get all edges pointing to a node.

        Args:
            node_id: Destination node ID.
            relation_type: Optional filter by relation type.

        Returns:
            List of matching incoming edges.
        """
        edges = [edge for edge in self._edges if edge.dst_id == node_id]
        if relation_type is not None:
            edges = [
                edge for edge in edges if edge.relation_type == relation_type
            ]
        return edges

    def get_edges_from(self, node_id: str, relation_type: str | None = None) -> list[Edge]:
        """Get all edges originating from a node.

        Args:
            node_id: Source node ID
            relation_type: Optional filter by edge type

        Returns:
            List of Edges
        """
        edges = [e for e in self._edges if e.src_id == node_id]
        if relation_type:
            edges = [e for e in edges if e.relation_type == relation_type]
        return edges

    def to_dict(self) -> dict:
        """Serialize scene graph to nested dict (JSON-ready).

        Returns:
            Dict with "nodes" and "edges" keys
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "edges": [edge.to_dict() for edge in self._edges],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        relationship_config_path: str | Path | None = None,
    ) -> SceneGraph:
        """Build a normalized graph from a Boxer scene-graph dictionary.

        Args:
            data: Boxer scene-graph dictionary containing objects and metadata.
            relationship_config_path: Optional explicit path to
                support_relationships.yaml. When omitted, the path is read from
                metadata.relationship_config.

        Returns:
            Normalized SceneGraph.

        Raises:
            FileNotFoundError: If the relationship YAML does not exist.
            ValueError: If the Boxer graph or relationship YAML is invalid.
        """
        metadata = data.get("metadata", {})
        if relationship_config_path is None and isinstance(metadata, dict):
            relationship_config_path = metadata.get("relationship_config")
        if relationship_config_path is None:
            raise ValueError(
                "Boxer scene graph metadata must provide 'relationship_config'"
            )
        static_furniture, null_classes = load_relationship_classes(
            relationship_config_path
        )

        objects = data.get("objects")
        if not isinstance(objects, list):
            raise ValueError("Expected Boxer scene graph with an 'objects' list")

        graph = cls()
        object_ids: dict[Any, str] = {}

        for boxer_object in objects:
            if not isinstance(boxer_object, dict):
                raise ValueError("Every Boxer object must be a JSON object")
            raw_id = boxer_object.get("id")
            label = boxer_object.get("label")
            if raw_id is None or not isinstance(label, str) or not label:
                raise ValueError("Every Boxer object requires 'id' and 'label'")
            if raw_id in object_ids:
                raise ValueError(f"Duplicate Boxer object id: {raw_id}")
            object_ids[raw_id] = f"{label}_{raw_id}"

        for boxer_object in objects:
            room_id = boxer_object.get("room_id")
            if not room_id or graph.get_node(str(room_id)):
                continue
            interior = boxer_object.get("room_interior_point")
            location = None
            if isinstance(interior, (list, tuple)) and len(interior) >= 2:
                location = [float(interior[0]), float(interior[1]), 0.0]
            room_type = boxer_object.get("room_type")
            graph.add_node(
                str(room_id),
                "room",
                {
                    "name": str(room_type or room_id).replace("_", " ").title(),
                    "room_type": room_type,
                    "location": location,
                },
            )

        for boxer_object in objects:
            raw_id = boxer_object["id"]
            label = boxer_object["label"]
            node_id = object_ids[raw_id]
            access_poses = cls._normalize_access_poses(
                boxer_object.get("base_access_poses", [])
            )
            attributes = dict(boxer_object)
            attributes.update(
                {
                    "name": f"{label.replace('_', ' ').title()} {raw_id}",
                    "label": label,
                    "boxer_id": raw_id,
                    "location": boxer_object.get("center"),
                    "access_poses": access_poses,
                    # Temporary compatibility until MoveToTarget uses access poses.
                    "visit_locations": [
                        [pose["x"], pose["y"], pose["yaw"]]
                        for pose in access_poses
                        if pose.get("valid", True)
                    ],
                }
            )
            graph.add_node(
                node_id,
                classify_boxer_label(label, static_furniture, null_classes),
                attributes,
            )

        for boxer_object in objects:
            node_id = object_ids[boxer_object["id"]]
            support_id = boxer_object.get("supported_by")
            if support_id is not None and support_id in object_ids:
                graph.add_edge(object_ids[support_id], node_id, "supports")
                continue
            room_id = boxer_object.get("room_id")
            if room_id and graph.get_node(str(room_id)):
                graph.add_edge(str(room_id), node_id, "contains")

        return graph

    @staticmethod
    def _normalize_access_poses(raw_poses: Any) -> list[dict[str, Any]]:
        """Normalize Boxer base access poses for navigation consumers.

        Args:
            raw_poses: Value of an object's base_access_poses field.

        Returns:
            Access-pose dictionaries sorted by clearance rank.
        """
        if not isinstance(raw_poses, list):
            return []

        normalized: list[dict[str, Any]] = []
        for raw_pose in raw_poses:
            if not isinstance(raw_pose, dict):
                continue
            position = raw_pose.get("position", {})
            orientation = raw_pose.get("orientation", {})
            if not isinstance(position, dict) or not isinstance(orientation, dict):
                continue
            if "x" not in position or "y" not in position:
                continue

            normalized.append(
                {
                    "pose_id": raw_pose.get("pose_id"),
                    "frame_id": raw_pose.get("frame_id", "map"),
                    "x": float(position["x"]),
                    "y": float(position["y"]),
                    "z": float(position.get("z", 0.0)),
                    "yaw": quaternion_to_yaw(orientation),
                    "valid": bool(raw_pose.get("valid", True)),
                    "rank": raw_pose.get("clearance_rank", 0),
                    "clearance_m": raw_pose.get("clearance_m"),
                    "distance_to_furniture_m": raw_pose.get(
                        "distance_to_furniture_m"
                    ),
                    "look_at_point": raw_pose.get("look_at_point"),
                    "map_status": raw_pose.get("map_status"),
                    "rejection_reasons": raw_pose.get("rejection_reasons", []),
                }
            )

        return sorted(normalized, key=lambda pose: pose.get("rank", 0))

    def to_json(self, path: str) -> None:
        """Save scene graph to JSON file.

        Args:
            path: File path to write to.

        Returns:
            None.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(
        cls,
        path: str,
        relationship_config_path: str | Path | None = None,
    ) -> SceneGraph:
        """Load and normalize a Boxer scene graph from JSON.

        Args:
            path: Path to the Boxer scene-graph JSON file.
            relationship_config_path: Optional explicit path to
                support_relationships.yaml. Relative metadata paths are
                resolved from the JSON file's directory.

        Returns:
            Loaded and normalized SceneGraph.

        Raises:
            FileNotFoundError: If the JSON or relationship YAML does not exist.
            ValueError: If the Boxer graph or relationship YAML is invalid.
        """
        with open(path, "r") as f:
            data = json.load(f)
        if relationship_config_path is None:
            metadata = data.get("metadata", {})
            configured_path = (
                metadata.get("relationship_config")
                if isinstance(metadata, dict)
                else None
            )
            if configured_path:
                configured_path = Path(configured_path).expanduser()
                if not configured_path.is_absolute():
                    configured_path = Path(path).resolve().parent / configured_path
                relationship_config_path = configured_path
        return cls.from_dict(data, relationship_config_path)

    def clear(self) -> None:
        """Clear all nodes and edges.

        Returns:
            None.
        """
        self.revision += 1
        self._nodes.clear()
        self._edges.clear()

    def __len__(self) -> int:
        """Get the number of nodes in the graph.

        Returns:
            Number of stored nodes.
        """
        return len(self._nodes)
