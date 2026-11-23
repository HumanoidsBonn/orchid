"""Robot manifest: declares robot capabilities and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModuleEndpoint:
    """Configuration for an external module (HTTP, ROS2, LLM, etc.)."""

    name: str
    endpoint: str
    type: str  # "http" | "llm" | "ros2_service" | "ros2_subscriber"
    config: dict = field(default_factory=dict)


@dataclass
class ControllerMap:
    """Maps controller roles to implementations."""

    nav: str | None = None  # e.g., "nav2"
    arm: str | None = None  # e.g., "moveit"
    head: str | None = None  # e.g., "hsrb_head"
    gripper: str | None = None  # e.g., "hsr_gripper"


@dataclass
class PlannerMap:
    """Maps planner roles/names to implementations."""

    manipulation: str | None = None
    default: str | None = None


@dataclass
class RobotManifest:
    """Robot capability declaration and hardware configuration."""

    robot_id: str
    robot_type: str
    capabilities: list[str] = field(default_factory=list)
    controllers: ControllerMap = field(default_factory=ControllerMap)
    planners: PlannerMap = field(default_factory=PlannerMap)
    hardware: dict = field(default_factory=dict)
    modules: list[ModuleEndpoint] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
