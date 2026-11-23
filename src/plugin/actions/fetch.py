"""Native Fetch subtree: semantic location selection, named-target scanning and pickup."""
from __future__ import annotations

import json
import math
import time
import urllib.error
from typing import Literal
import uuid
import py_trees
from src.core.action import AbstractAction, ActionResult, AsyncAction
from src.core.failure import FailureCode, FailureInfo, FailureOrigin, MotionFailure
from src.core.llm_planner import LLMPlanner
from src.core.scene_graph_query import SceneGraphQuery
from src.tree.bt_node_wrapper import ActionScope, BtConditionLeaf, retry_eligible, terminal_failure
from src.workflows.catalog import visitable_locations
from src.plugin.actions._helpers import scene_location_description
import src.core.logger as log


def location_choices(context):
    return [{"value": row["id"], "label": row["label"]} for row in visitable_locations(context)]


def choose_search_locations(context, target, extra_count, response):
    """Validate semantic matches, then deterministically rank known visit locations."""
    if not isinstance(response, dict) or set(response) - {"known_object_ids", "additional_furniture_ids"}:
        raise ValueError("Invalid search-location response")
    known, additional = response.get("known_object_ids", []), response.get("additional_furniture_ids", [])
    if not isinstance(known, list) or not isinstance(additional, list) or not all(isinstance(x, str) for x in known + additional):
        raise ValueError("Search locations must be lists of scene IDs")
    allowed = {row["id"] for row in visitable_locations(context)}
    query = SceneGraphQuery(context.scene_graph)
    locations = []
    for object_id in known:
        node = context.scene_graph.get_node(object_id)
        if node is None or node.type != "object":
            raise ValueError(f"Unknown movable-object ID: {object_id}")
        furniture = query.get_supporting_furniture(object_id)
        if furniture not in allowed:
            raise ValueError(f"Object has no visitable supporting furniture: {object_id}")
        if furniture not in locations:
            locations.append(furniture)
    pose = context.require_nav().get_current_pose()
    def distance(node_id):
        visit = query.get_nearest_visit_location(node_id, [pose.x, pose.y])
        return math.hypot(visit["x"] - pose.x, visit["y"] - pose.y)
    locations.sort(key=lambda node_id: (distance(node_id), node_id))
    extras = []
    for node_id in additional:
        if node_id not in allowed:
            raise ValueError(f"Not a visitable furniture ID: {node_id}")
        if node_id not in locations and node_id not in extras:
            extras.append(node_id)
    return locations + extras[:extra_count]


class ResolveFetchLocations(AsyncAction):
    """A bounded semantic query, not an LLM-generated executable tree."""
    def __init__(self, target: str, output_key: str, kind: Literal["search", "hint", "delivery"] = "search",
                 response_key: str = "", extra_search_locations: int = 5, max_attempts: int = 3, target_label: str | None = None):
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.target_label = target_label
        self.max_attempts = max_attempts
        self.target, self.output_key, self.kind = target, output_key, kind
        self.response_key, self.extra_search_locations = response_key, extra_search_locations
        super().__init__()
        self._generation = 0
        self._result = None

    def build_node(self, context, name):
        from src.tree.bt_node_wrapper import retry_eligible
        leaf = super().build_node(context, name)
        retry = py_trees.decorators.Retry(name=name+" network retry", child=leaf, num_failures=self.max_attempts)
        return py_trees.decorators.EternalGuard(name=name+" retry eligibility", child=retry,
            condition=lambda: retry_eligible(context, leaf, retry))

    def start(self, context):
        self._cancel_requested = False
        self._generation += 1
        generation = self._generation
        self._result = None
        response = str(context.world_state.get(self.response_key) or "").strip() if self.response_key else ""
        # Capture graph/input data before scheduling; all execution writes are guarded.
        locations = visitable_locations(context)
        if self.kind == 'search':
            log.info('Fetch ranking %d visitable furniture locations; additional search limit=%d',
                     len(locations), self.extra_search_locations)
        allowed = {row["id"] for row in locations}
        known_objects = []
        if context.scene_graph is not None:
            query = SceneGraphQuery(context.scene_graph)
            known_objects = [{"id": n.id, "label": n.attributes.get("label", n.id), "furniture_id": query.get_supporting_furniture(n.id)}
                             for n in context.scene_graph.get_nodes("object")
                             if query.get_supporting_furniture(n.id) in allowed]
        def current():
            return generation == self._generation and not self._cancel_requested and not context.execution.cancel_requested and context.execution.control is None
        def work():
            if self.kind != "search":
                if response.casefold() in {"here", "bring it here", "me", "to me", ""}:
                    return None
                if response in allowed:
                    return response
                matches = [row["id"] for row in locations if response.casefold() == row["label"].casefold()]
                if len(matches) == 1:
                    return matches[0]
            module = LLMPlanner(context)._resolve_llm_module_name()
            system = ("Match the user's named item to known object IDs and rank likely furniture search locations. "
                      "Use only the supplied exact IDs. Furniture must have visit poses. Respect room context. "
                      "Return JSON only: {\"known_object_ids\":[],\"additional_furniture_ids\":[]}. "
                      "Include every matching known object. Additional furniture is ordered by likelihood, without duplicates.") if self.kind == "search" else (
                      "Resolve the user's location answer to exactly one supplied furniture ID, using room context. "
                      "Return JSON only: {\"location_id\":null} when ambiguous, otherwise {\"location_id\":\"exact ID\"}. Never invent IDs.")
            payload = {"target": self.target, "answer": response, "furniture_with_visit_poses": locations}
            if self.kind == "search":
                payload["known_objects"] = known_objects
                payload["additional_limit"] = self.extra_search_locations
            if not current():
                return None
            log.speak("Planning")
            try:
                result = context.modules.call(module, {"system_prompt": system, "prompt": json.dumps(payload)})
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
                raise MotionFailure(str(exc), FailureInfo(FailureCode.NETWORK_ERROR,
                    retryable=retryable, backend_code=getattr(exc, "code", None))) from exc
            if isinstance(result, dict) and "error" in result:
                retryable = result.get("retryable") is True and result.get("error_kind") in {"network", "timeout", "http"}
                raise MotionFailure(str(result["error"]), FailureInfo(FailureCode.NETWORK_ERROR,
                    retryable=retryable, backend_code=result.get("status_code")))
            content = LLMPlanner(context)._extract_llm_content(result, module)
            data = json.loads(content.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            if self.kind == "search":
                selected = choose_search_locations(context, self.target, self.extra_search_locations, data)
                query = SceneGraphQuery(context.scene_graph)
                known = {query.get_supporting_furniture(node_id) for node_id in data.get("known_object_ids",[])}
                return {"locations":selected,"known_locations":[node_id for node_id in selected if node_id in known],"known_object_ids":data.get("known_object_ids",[])}
            if not isinstance(data, dict) or set(data) != {"location_id"}:
                raise ValueError("Invalid location answer response")
            value = data["location_id"]
            if value is not None and value not in allowed:
                raise ValueError("Location resolver returned an ineligible ID")
            return value
        def finish(value):
            if not current():
                return
            try:
                if self.kind == "search":
                    known_locations = value["known_locations"]
                    context.world_state.set(self.output_key+".known_locations",known_locations)
                    from ._helpers import scene_node_reference
                    context.world_state.set("scene_known_objects",[ref for node_id in value["known_object_ids"] if (ref:=scene_node_reference(context,node_id)) is not None])
                    value = value["locations"]
                    eligible = {row["id"] for row in visitable_locations(context)}
                    if any(node_id not in eligible for node_id in value):
                        raise ValueError("Search location is no longer visitable")
                elif value is not None and value not in {row["id"] for row in visitable_locations(context)}:
                    raise ValueError("Location is no longer visitable")
                context.world_state.set(self.output_key, value)
                if self.kind == 'search':
                    labels = {row['id']: row['label'] for row in visitable_locations(context)}
                    log.info('Fetch selected search order (%d locations): %s', len(value),
                             ' -> '.join(labels[node_id] for node_id in value) or '(none)')
                    if known_locations:
                        places = []
                        for node_id in known_locations:
                            furniture,room = scene_location_description(context,node_id)
                            places.append(f"on {furniture} in room {room}" if room else f"on {furniture}")
                        log.speak("Found the last known %s of %s %s","location" if len(places)==1 else "locations",(self.target_label or self.target.replace("_"," ")),"; and ".join(places))
                self._result = ActionResult(status="success", payload={"resolved": value})
            except Exception as exc:
                fail(exc)
        def fail(exc):
            if current():
                log.warn("Fetch location resolution failed: %s", exc)
                self._result = ActionResult(status="failure", reason=str(exc), failure=getattr(exc, "failure",
                    FailureInfo(FailureCode.INVALID_REQUEST if isinstance(exc, ValueError) else FailureCode.UNKNOWN)))
        context.run_in_context_loop(work, callback=finish, error_callback=fail)

    def check_result(self, context):
        return self._result or ActionResult(status="running")

    def cancel(self, context):
        self._cancel_requested = True
        self._generation += 1


class FetchObject(AbstractAction):
    """Fetch a named item with native search, pickup, destination input and release."""
    def __init__(self, target: str, extra_search_locations: int = 5, target_label: str | None = None):
        if not target.strip() or not 0 <= extra_search_locations <= 20:
            raise ValueError("Fetch requires a named target and 0–20 additional search locations")
        self.target, self.extra_search_locations = target, extra_search_locations
        self.target_label = target_label
        self._key = "fetch." + uuid.uuid4().hex
        self._fatal = None
        self._found = False
        self._visited = set()

    def build_node(self, context, name):
        from src.plugin.actions.navigation import MoveToTarget
        from src.plugin.actions.observe import ScanScene, ObserveScene
        from src.plugin.actions.user_input import AskUser
        from src.plugin.actions.grasp_alignment import EstimateSimplePcaGrasp
        from src.plugin.actions.hsr_pickup import HsrPickup
        from src.plugin.actions.manipulation import GraspObjectAction
        search_key, hint_key, destination_key = (self._key + suffix for suffix in (".search", ".hint", ".destination"))
        answer_key = self._key + ".answer"
        def sequence(label, children):
            return py_trees.composites.Sequence(name=label, memory=True, children=children)
        def succeeded(ctx, status):
            return ActionResult(status="success" if status == py_trees.common.Status.SUCCESS else "failure")
        def result_of(root):
            cause = getattr(root.tip(), "result", None)
            if cause is not None and cause.status != "success":
                return cause
            return terminal_failure(context, root) or ActionResult(status="failure", failure=FailureInfo())
        attempts = []
        def label_search_locations(ctx):
            selected = ctx.world_state.get(search_key) or []
            labels = {row['id']: row['label'] for row in visitable_locations(ctx)}
            automatic.name = f'Selected search locations ({len(selected)})'
            for slot, attempt in enumerate(attempts):
                label = (f'Search: {labels.get(selected[slot], selected[slot])}' if slot < len(selected)
                         else 'Unused search slot')
                attempt.name = label
                attempt.binding.name = label
                attempt.decorated.name = label
        # One slot per possible known support plus bounded additional suggestions.
        for index in range(len(visitable_locations(context)) + self.extra_search_locations):
            navigation = MoveToTarget(prefer_nearest=True)
            scan_node = ScanScene(target=self.target, target_label=self.target_label, exploration_mode=True, strict_failures=False).build_node(context, "Scan for target")
            observation_node = ObserveScene(target=self.target, target_label=self.target_label, replan_on_failure=False).build_node(context, "Observe found target")
            operations = sequence(f"Search visit {index + 1}", [navigation.build_node(context, "Navigate to search location"),
                scan_node, observation_node])
            def enter(ctx, index=index, navigation=navigation):
                if index == 0:
                    label_search_locations(ctx)
                if self._fatal:
                    return self._fatal
                locations = ctx.world_state.get(search_key) or []
                if index >= len(locations) or locations[index] in self._visited:
                    return ActionResult(status="failure", reason="No unvisited candidate in slot", failure=FailureInfo(FailureCode.NOT_FOUND))
                navigation.target = locations[index]
                ctx.execution.stage_labels[attempts[index].id] = ('Verifying at known location'
                    if navigation.target in (ctx.world_state.get(search_key+'.known_locations') or []) else 'Exploring')
                if navigation.target not in {row["id"] for row in visitable_locations(ctx)}:
                    self._fatal = ActionResult(status="failure", reason="Search location no longer visitable", failure=FailureInfo(FailureCode.INVALID_STATE))
                    return self._fatal
                self._visited.add(navigation.target)
            def complete(ctx, status, operations=operations, index=index, navigation=navigation, scan_node=scan_node, observation_node=observation_node):
                if status == py_trees.common.Status.SUCCESS:
                    self._found = True
                    ctx.world_state.set("scene_known_objects",[])
                    return ActionResult(status="success")
                result = result_of(operations)
                failed_observation = next((node for node in (scan_node, observation_node) if node.status == py_trees.common.Status.FAILURE), None)
                if failed_observation is not None and failed_observation.result is not None:
                    result = failed_observation.result
                motion_failed = result.failure is not None and result.failure.origin == FailureOrigin.CONTROLLER
                if result.failure and result.failure.code == FailureCode.CANCELLED:
                    self._fatal = result
                elif failed_observation is not None and result.status == "failure" and (not result.failure or result.failure.code != FailureCode.NOT_FOUND):
                    log.warn("Fetch skipping search location %s after %s failed: %s; failure=%s",
                             navigation.target, failed_observation.name, result.reason, result.failure)
                elif motion_failed and result.status == "failure" and (not result.failure or result.failure.code not in {FailureCode.CANCELLED, FailureCode.INVALID_REQUEST}):
                    log.warn("Fetch skipping search location after motion failure: %s", result.reason)
                elif not result.failure or result.failure.code != FailureCode.NOT_FOUND:
                    self._fatal = result
                else:
                    query=SceneGraphQuery(ctx.scene_graph)
                    ctx.world_state.set("scene_known_objects",[ref for ref in (ctx.world_state.get("scene_known_objects") or []) if query.get_supporting_furniture(ref["id"])!=navigation.target])
                    eligible = {row["id"] for row in visitable_locations(ctx)}
                    remaining = (ctx.world_state.get(search_key) or [])[index+1:]
                    if any(location not in self._visited and location in eligible for location in remaining):
                        log.speak("Continuing to look for %s",(self.target_label or self.target.replace("_"," ")))
                return result
            attempts.append(ActionScope(self, context, f"Awaiting search ranking: slot {index + 1}", operations, on_enter=enter, on_complete=complete, on_cancel=lambda ctx: None))
        automatic = py_trees.composites.Selector(name="Known locations, then likely furniture", memory=True, children=attempts)
        ask_hint = AskUser(prompt="Where should I look for {}?".format(self.target_label or self.target.replace("_", " ")), response_key=answer_key, replan_on_response=False)
        hint_resolver = ResolveFetchLocations(self.target, hint_key, "hint", answer_key, target_label=self.target_label)
        hint_ops = sequence("Ask for another search location", [ask_hint.build_node(context, "Ask where to find it"), hint_resolver.build_node(context, "Resolve search hint")])
        def hint_enter(ctx):
            ask_hint.choices = location_choices(ctx)
            if not ask_hint.choices:
                return ActionResult(status="failure", reason="No visitable search locations remain", failure=FailureInfo(FailureCode.INVALID_STATE))
        def hint_complete(ctx, status):
            if status != py_trees.common.Status.SUCCESS:
                self._fatal = result_of(hint_ops)
                return self._fatal
            if status == py_trees.common.Status.SUCCESS:
                value = ctx.world_state.get(hint_key)
                if value:
                    self._visited.discard(value)  # Explicit user instruction permits revisiting.
                ctx.world_state.set(search_key, [value] if value else [])
                ctx.world_state.set(search_key+'.known_locations',[])
            return succeeded(ctx, status)
        hint_scope = ActionScope(self, context, "Search hint", hint_ops, on_enter=hint_enter, on_complete=hint_complete, on_cancel=lambda ctx: None)
        guarded_hint = py_trees.decorators.EternalGuard(name="No search backend failure", child=hint_scope, condition=lambda: self._fatal is None)
        round_node = py_trees.composites.Selector(name="Search or request a location", memory=True, children=[automatic, guarded_hint])
        search = py_trees.composites.Selector(name="Find named target", memory=False, children=[
            BtConditionLeaf(lambda ctx:self._found, context, "Target observed"),
            py_trees.decorators.SuccessIsRunning(name="Continue user-guided search", child=round_node)])
        guarded_search = py_trees.decorators.EternalGuard(name="Search failure guard", child=search, condition=lambda:self._fatal is None)
        destination = AskUser(prompt="Where should I bring it? Choose a named location.", response_key=answer_key, replan_on_response=False)
        resolve_destination = ResolveFetchLocations(self.target, destination_key, "delivery", answer_key, target_label=self.target_label)
        ask_delivery_ops = sequence("Choose delivery location", [destination.build_node(context, "Ask where to bring it"), resolve_destination.build_node(context, "Resolve delivery location")])
        def delivery_question_enter(ctx):
            destination.choices = location_choices(ctx)
            if not destination.choices:
                return ActionResult(status="failure", reason="No visitable delivery locations remain", failure=FailureInfo(FailureCode.INVALID_STATE))
        def delivery_question_complete(ctx, status):
            if status != py_trees.common.Status.SUCCESS:
                self._fatal = result_of(ask_delivery_ops)
                return self._fatal
            return ActionResult(status="success")
        ask_delivery = ActionScope(self, context, "Delivery question", ask_delivery_ops,
            on_enter=delivery_question_enter, on_complete=delivery_question_complete, on_cancel=lambda ctx:None)
        choose_destination = py_trees.composites.Selector(name="Resolve a named destination", memory=False, children=[
            BtConditionLeaf(lambda ctx:bool(ctx.world_state.get(destination_key)), context, "Destination selected"),
            py_trees.decorators.SuccessIsRunning(name="Clarify missing or ambiguous destination", child=ask_delivery)])
        deliver = MoveToTarget(prefer_nearest=True)
        delivery_ops = deliver.build_node(context, "Navigate to delivery location")
        def bind_delivery(ctx):
            deliver.target = ctx.world_state.get(destination_key)
            if deliver.target not in {row["id"] for row in visitable_locations(ctx)}:
                return ActionResult(status="failure", reason="Delivery location no longer visitable", failure=FailureInfo(FailureCode.INVALID_STATE))
        delivery = ActionScope(self, context, "Delivery", delivery_ops, on_enter=bind_delivery, on_complete=succeeded, on_cancel=lambda ctx:None)
        close_leaf=GraspObjectAction().build_node(context,"Close gripper before Fetch")
        close_retry=py_trees.decorators.Retry(name="Retry initial gripper close",child=close_leaf,num_failures=2)
        close=py_trees.decorators.EternalGuard(name="Initial gripper close retry eligibility",child=close_retry,
            condition=lambda:retry_eligible(context,close_leaf,close_retry))
        ranking=ResolveFetchLocations(self.target, search_key, extra_search_locations=self.extra_search_locations, target_label=self.target_label).build_node(context, "Select search locations")
        estimate=EstimateSimplePcaGrasp(target=self.target).build_node(context, "Estimate simple grasp")
        pickup=HsrPickup(target=self.target, target_label=self.target_label).build_node(context, "Pick up target")
        for node,label in ((close,'Preparing gripper'),(ranking,'Planning for locations'),
                (hint_scope,'Asking where to look'),(estimate,'Estimating Grasp'),(pickup,'Performing pickup'),
                (choose_destination,'Asking for drop'),(delivery,'Going to drop')):
            context.execution.stage_labels[node.id]=label
        root = sequence("Fetch operations", [close,ranking,guarded_search,estimate,pickup,choose_destination,delivery])
        return ActionScope(self, context, name, root)

    def on_enter(self, context):
        self._fatal, self._found, self._visited = None, False, set()
        context.world_state.set("scene_known_objects",[])
        for suffix in (".search", ".hint", ".destination", ".answer"):
            context.world_state.set(self._key + suffix, None)
        context.world_state.set(self._key+'.search.known_locations',[])
        for suffix in ("_visible", "_scan_bbox", "_scan_view"):
            context.world_state.set(self.target + suffix, None)

    def on_complete(self, context, status):
        context.world_state.set("scene_known_objects",[])
        if self._fatal:
            return self._fatal
        return ActionResult(status="success" if status == py_trees.common.Status.SUCCESS else "failure")

    def on_cancel(self,context):
        context.world_state.set("scene_known_objects",[])
