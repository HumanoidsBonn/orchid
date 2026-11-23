"""Tests for scene graph loading and context integration."""

from __future__ import annotations

import pytest

from src.core.scene_graph import SceneGraph
from src.orchestrator.session import RobotSession
from tests.fixtures.mock_manifests import basic_registry, rich_manifest


def test_scene_graph_loads_from_json(rich_manifest, basic_registry):
    """Test that scene graph loads from example_scene_graph.json."""
    session = RobotSession(rich_manifest, basic_registry)
    
    # Context should have scene graph loaded
    assert session.context.scene_graph is not None
    assert isinstance(session.context.scene_graph, SceneGraph)


def test_scene_graph_contains_expected_rooms(rich_manifest, basic_registry):
    """Test that scene graph contains the three main rooms."""
    session = RobotSession(rich_manifest, basic_registry)
    scene_graph = session.context.scene_graph
    
    assert scene_graph is not None
    kitchen = scene_graph.get_node("kitchen")
    living_room = scene_graph.get_node("living_room")
    bedroom = scene_graph.get_node("bedroom")
    
    assert kitchen is not None
    assert living_room is not None
    assert bedroom is not None
    
    assert kitchen.type == "room"
    assert living_room.type == "room"
    assert bedroom.type == "room"


def test_scene_graph_contains_objects_and_relationships(rich_manifest, basic_registry):
    """Test that scene graph contains objects and their relationships."""
    session = RobotSession(rich_manifest, basic_registry)
    scene_graph = session.context.scene_graph
    
    assert scene_graph is not None
    
    # Check for specific objects
    mug = scene_graph.get_node("mug")
    bed = scene_graph.get_node("bed")
    sofa = scene_graph.get_node("sofa")
    
    assert mug is not None
    assert mug.type == "object"
    assert mug.attributes.get("name") == "Coffee Mug"
    
    assert bed is not None
    assert bed.type == "furniture"
    
    assert sofa is not None
    assert sofa.type == "furniture"


def test_scene_graph_relationships_kitchen(rich_manifest, basic_registry):
    """Test relationships in kitchen scene."""
    session = RobotSession(rich_manifest, basic_registry)
    scene_graph = session.context.scene_graph
    
    assert scene_graph is not None
    
    # Check that kitchen contains items
    kitchen_neighbors = scene_graph.get_neighbors("kitchen", relation_type="contains")
    kitchen_items = [n.id for n in kitchen_neighbors]
    
    assert "kitchen_counter" in kitchen_items or "kitchen_table" in kitchen_items or "refrigerator" in kitchen_items


def test_scene_graph_relationships_bedroom(rich_manifest, basic_registry):
    """Test relationships in bedroom scene."""
    session = RobotSession(rich_manifest, basic_registry)
    scene_graph = session.context.scene_graph
    
    assert scene_graph is not None
    
    # Check that bedroom contains items
    bedroom_neighbors = scene_graph.get_neighbors("bedroom", relation_type="contains")
    bedroom_items = [n.id for n in bedroom_neighbors]
    
    assert "bed" in bedroom_items or "wardrobe" in bedroom_items or "nightstand" in bedroom_items


def test_scene_graph_serialization(rich_manifest, basic_registry):
    """Test that scene graph can be serialized and deserialized."""
    session = RobotSession(rich_manifest, basic_registry)
    scene_graph = session.context.scene_graph
    
    assert scene_graph is not None
    
    # Serialize to dict
    graph_dict = scene_graph.to_dict()
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert len(graph_dict["nodes"]) > 0
    assert len(graph_dict["edges"]) > 0
    
    # Deserialize from dict
    reconstructed_graph = SceneGraph.from_dict(graph_dict)
    assert reconstructed_graph.get_node("kitchen") is not None
    assert reconstructed_graph.get_node("mug") is not None
    assert len(reconstructed_graph.get_neighbors("tabletop", relation_type="on")) > 0
