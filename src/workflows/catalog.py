"""Small, revisioned workflow metadata and validated Plan factories."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from src.core.scene_graph_query import SceneGraphQuery
from src.tree.plan import ActionNode, CompositeNode, DecoratorNode, Plan

SCHEMA_VERSION = 1


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    query: str = Field(min_length=1)


class WorkflowSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: list[str] = Field(default_factory=lambda: ["release", "close_gripper", "home", "fetch"])
    items: list[Item] = Field(default_factory=list)
    include_scene_items: bool = False
    extra_search_locations: int = Field(default=5, ge=0, le=20)


class NoInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FetchInputs(NoInputs):
    item_id: str = Field(min_length=1, title="Object", json_schema_extra={"choices":"items"})


@dataclass(frozen=True)
class WorkflowRequest:
    workflow_id: str
    inputs: dict[str, Any]
    display_name: str = ""

    def __str__(self):
        if self.display_name:
            return self.display_name
        return f"{self.workflow_id}: {self.inputs['item_id']}" if self.workflow_id == "fetch" else self.workflow_id.title()


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    title: str
    required_actions: tuple[str, ...]
    required_controllers: tuple[str, ...]
    factory: Callable
    inputs: type[BaseModel] = NoInputs


def _action(key, name, **params):
    return ActionNode(action_type=key, name=name, params=params)


def _release(catalog, inputs):
    return DecoratorNode(type="retry",name="Retry failed release",params={"num_attempts":2},
                         child=_action("ReleaseObjectAction", "Release"))


def _close_gripper(catalog, inputs):
    return DecoratorNode(type="retry", name="Retry failed gripper close", params={"num_attempts":2},
                         child=_action("GraspObjectAction", "Close gripper"))


def _home(catalog, inputs):
    return CompositeNode(type="sequence", name="Home", children=[_action("MoveArmToState", "Arm to go", state="go"), _action("MoveHeadToState", "Center head", state="center")])


def _fetch(catalog, inputs):
    item = next((item for item in catalog.items() if item["id"] == inputs["item_id"]), None)
    if item is None:
        raise ValueError("Selected Fetch item is no longer available")
    return _action("FetchObject", f"Fetch {item['label']}", target=item["query"], target_label=item["label"], extra_search_locations=catalog.settings.extra_search_locations)


DEFINITIONS = {
    "release": WorkflowDefinition("release", "Release", ("ReleaseObjectAction",), ("gripper",), _release),
    "close_gripper": WorkflowDefinition("close_gripper", "Close gripper", ("GraspObjectAction",), ("gripper",), _close_gripper),
    "home": WorkflowDefinition("home", "Home", ("MoveArmToState", "MoveHeadToState"), ("arm", "head"), _home),
    "fetch": WorkflowDefinition("fetch", "Fetch", ("FetchObject", "GraspObjectAction", "MoveToTarget", "ScanScene", "ObserveScene", "EstimateSimplePcaGrasp", "HsrPickup", "AskUser"), ("nav", "arm", "head", "gripper"), _fetch, FetchInputs),
}


def visitable_locations(context) -> list[dict]:
    graph = context.scene_graph
    if graph is None:
        return []
    query = SceneGraphQuery(graph)
    result = []
    for node in sorted(graph.get_nodes("furniture"), key=lambda n: n.id):
        poses = query.get_visit_locations(node.id)
        if not poses or query.get_best_visit_location(node.id) is None:
            continue
        room_id = query.get_containing_room(node.id)
        room = graph.get_node(room_id) if room_id else None
        room_label = str(room.attributes.get("room_type") or room_id) if room else "Unassigned"
        label = str(node.attributes.get("label") or node.attributes.get("name") or node.id)
        result.append({"id": node.id, "label": f"{label} ({node.id}) — {room_label}", "room_id": room_id, "room": room_label})
    return result


class WorkflowCatalog:
    def __init__(self, context, robot_id: str, settings: WorkflowSettings | None = None):
        self.context, self.robot_id = context, robot_id
        if settings is None:
            default = Path(__file__).resolve().parents[2] / "config" / "workflows.json"
            path = Path(os.getenv("ORCHID_WORKFLOWS_CONFIG", str(default)))
            settings = WorkflowSettings.model_validate(json.loads(path.read_text()))
        unknown = set(settings.enabled) - DEFINITIONS.keys()
        if unknown:
            raise ValueError(f"Unknown workflows: {sorted(unknown)}")
        self.settings = settings

    def items(self) -> list[dict]:
        def item_key(value):
            return re.sub(r'[_\s]+', ' ', value).strip().casefold()
        result = {item_key(item.query): item.model_dump() for item in self.settings.items}
        if self.settings.include_scene_items and self.context.scene_graph is not None:
            for node in sorted(self.context.scene_graph.get_nodes("object"), key=lambda n: n.id):
                if node.attributes.get("movable") is False:
                    continue
                label = str(node.attributes.get("label") or node.attributes.get("name") or node.id).strip()
                result.setdefault(item_key(label), {"id": f"scene:{node.id}", "label": label, "query": label})
        return sorted(result.values(), key=lambda item: (item["label"].casefold(), item["id"]))

    def unavailable_reason(self, definition: WorkflowDefinition) -> str:
        missing = [key for key in definition.required_actions if key not in self.context.supported_actions]
        missing += [key for key in definition.required_controllers if getattr(self.context, key, None) is None]
        if definition.id == "fetch":
            if self.context.scene_graph is None:
                missing.append('scene graph not loaded')
            elif not visitable_locations(self.context):
                missing.append('valid furniture access poses in loaded scene graph')
        return "Missing: " + ", ".join(missing) if missing else ""

    def document(self) -> dict:
        workflows = []
        for key in self.settings.enabled:
            definition = DEFINITIONS[key]
            reason = self.unavailable_reason(definition)
            schema = definition.inputs.model_json_schema()
            fields = []
            for name, prop in schema.get("properties", {}).items():
                kind = "select" if "choices" in prop or "enum" in prop else {"integer":"number", "number":"number", "boolean":"checkbox"}.get(prop.get("type"), "text")
                field = {"name":name, "label":prop.get("title",name), "type":kind, "required":name in schema.get("required",[])}
                for source, dest in (("choices","choices"),("enum","enum"),("default","default"),("minimum","min"),("maximum","max")):
                    if source in prop: field[dest] = prop[source]
                fields.append(field)
            workflows.append({"id": key, "title": definition.title, "fields": fields, "available": not reason, "reason": reason})
        content = {"schema_version": SCHEMA_VERSION, "robot_id": self.robot_id, "workflows": workflows,
                   "items": self.items(), "locations": visitable_locations(self.context)}
        content["revision"] = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return content

    def validate(self, workflow_id: str, inputs: dict) -> WorkflowRequest:
        if workflow_id not in self.settings.enabled or workflow_id not in DEFINITIONS:
            raise ValueError(f"Workflow not enabled: {workflow_id}")
        definition = DEFINITIONS[workflow_id]
        reason = self.unavailable_reason(definition)
        if reason:
            raise ValueError(reason)
        validated = definition.inputs.model_validate(inputs).model_dump()
        display_name = definition.title
        if workflow_id == "fetch":
            item = next((item for item in self.items() if item["id"] == validated["item_id"]), None)
            if item is None:
                raise ValueError("Selected Fetch item is no longer available")
            display_name = f"{definition.title} {item['label']}"
        return WorkflowRequest(workflow_id, validated, display_name)

    def build_plan(self, request: WorkflowRequest) -> Plan:
        request = self.validate(request.workflow_id, request.inputs)
        root = DEFINITIONS[request.workflow_id].factory(self, request.inputs)
        return Plan(goal=str(request), root=root)
