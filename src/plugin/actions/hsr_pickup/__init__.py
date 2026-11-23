"""HSR pickup native subtree and asynchronous planning operation."""
from __future__ import annotations
from typing import Literal
import numpy as np
import py_trees
import src.core.logger as log
from src.core.action import AbstractAction, AsyncAction, ActionResult
from src.core.failure import FailureCode, FailureInfo
from src.tree.bt_node_wrapper import ActionScope, BtConditionLeaf, retry_eligible, terminal_failure
from .model import ApproachMode, BackoffDirection, TargetCollisionGeometry, ARM_JOINTS, BASE_JOINTS, PickupBackendSettings, PickupPlan, PickupRequest
from .planning import PickupPlanningClient, pickup_observation


class PlanHsrPickup(AsyncAction):
    """Plan a complete collision-aware pickup without executing or writing the scene.

    auto tries the native server's configured mode priority. Results include
    native staging order and connected prepared trajectories through undock.
    The existing named-arm action plans stowing from measured state.
    """
    def __init__(self, target: str, approach_mode: ApproachMode = "auto",
                 backoff_direction: BackoffDirection = "grasp_axis", pregrasp_distance_m: float = .25,
                 min_insertion_m: float = .04, max_insertion_m: float = .60,
                 clearance_m: float = .03, post_grasp_lift_m: float = .05,
                 collision_hull_inflation_factor: float = 2., max_candidates: int = 8,
                 output_key: str | None = None, debug_visualization: bool = True,
                 open_width_m: float = .15, diagnostic_directory: str | None = None,
                 target_collision_geometry: TargetCollisionGeometry = "mesh", target_label: str | None = None):
        super().__init__()
        self.target, self.approach_mode = target, approach_mode
        self.target_label = target_label
        self.backoff_direction, self.pregrasp_distance_m = backoff_direction, pregrasp_distance_m
        self.min_insertion_m, self.max_insertion_m = min_insertion_m, max_insertion_m
        self.clearance_m, self.post_grasp_lift_m = clearance_m, post_grasp_lift_m
        self.collision_hull_inflation_factor, self.max_candidates = collision_hull_inflation_factor, max_candidates
        self.output_key, self.debug_visualization, self.open_width_m = output_key, debug_visualization, open_width_m
        self.diagnostic_directory = diagnostic_directory
        self.target_collision_geometry = target_collision_geometry
        self._generation = 0
        self._result = None
        self._client = None

    def start(self, context):
        self._generation+=1
        generation=self._generation
        self._cancel_requested=False
        self._result=None
        def current():
            return (generation==self._generation and not self._cancel_requested and
                    context.execution.control is None and not context.execution.cancel_requested)
        try:
            request=PickupRequest(target=self.target,approach_mode=self.approach_mode,
                backoff_direction=self.backoff_direction,pregrasp_distance_m=self.pregrasp_distance_m,
                min_insertion_m=self.min_insertion_m,max_insertion_m=self.max_insertion_m,
                clearance_m=self.clearance_m,post_grasp_lift_m=self.post_grasp_lift_m,
                collision_hull_inflation_factor=self.collision_hull_inflation_factor,
                max_candidates=self.max_candidates,open_width_m=self.open_width_m,
                target_collision_geometry=self.target_collision_geometry)
            observation=pickup_observation(context,self.target)
            settings=PickupBackendSettings.from_hardware(context.hardware)
            self._client=self._client or PickupPlanningClient(context.ros2_node,settings)
            key=self.output_key or f"{self.target}_pickup_plan"
            context.world_state.set(key,None)
            context.world_state.set(f"{key}.undock_pose",None)
            context.world_state.set(f"{key}.retreat_mode",None)
        except Exception as exc:
            self._result=ActionResult(status="failure",reason=str(exc),failure=FailureInfo(FailureCode.INVALID_REQUEST))
            log.warn("Pickup planning input failed: %s",exc)
            return
        def work():
            if not current():
                return None
            log.speak("Planning to pick up %s",(self.target_label or self.target.replace("_"," ")))
            inputs=self._client.collect_inputs(context,request.target,request.collision_hull_inflation_factor,observation,
                target_collision_geometry=request.target_collision_geometry)
            plan=self._client.plan(inputs,request,cancelled=lambda:not current())
            if self.diagnostic_directory and current():
                import json
                from pathlib import Path
                from .diagnostics import export_inputs,plan_report
                export_inputs(inputs,self.diagnostic_directory,request.target)
                (Path(self.diagnostic_directory)/"result.json").write_text(json.dumps(plan_report(plan),indent=2))
                log.info("Pickup diagnostic export: %s",self.diagnostic_directory)
            return plan
        def finish(plan):
            if not current():
                return
            context.world_state.set(key,plan)
            if self.debug_visualization:
                try:
                    self._client.publish(plan)
                except Exception as exc:
                    log.warn("Pickup RViz publication failed (planning result retained): %s",exc)
            if plan.motions:
                context.world_state.set(f"{key}.mode",plan.selected_mode)
                context.world_state.set(f"{key}.staging_order",plan.staging_order)
                for motion in plan.motions:
                    context.world_state.set(f"{key}.{motion.name}",motion.trajectory)
                # Save the base position used as the actual invocation start.
                pose={k:plan.inputs.start_joints[n] for k,n in zip(("x","y","yaw"),BASE_JOINTS)}
                pose["frame_id"]=plan.inputs.frame_id
                context.world_state.set(f"{self.target}_pick_start_base_pose",pose)
                # Use the native planner's selected farther pose for live undocking.
                # Keep the original pick-start metadata intact for other workflows.
                undock=next((motion for motion in plan.motions if motion.name=="undock"),None)
                if undock is not None:
                    undock_pose={k:undock.end_joints[n] for k,n in zip(("x","y","yaw"),BASE_JOINTS)}
                    undock_pose["frame_id"]=plan.inputs.frame_id
                    context.world_state.set(f"{key}.undock_pose",undock_pose)
                    log.info("Pickup undock selected pose=%s",undock_pose)
            self._result=ActionResult(status="success" if plan.motions else "failure",reason=plan.diagnostics["reason"],
                failure=None if plan.motions else FailureInfo(FailureCode.PLANNING_FAILED),
                payload={"plan_key":key,"selected_mode":plan.selected_mode,"staging_order":plan.staging_order,"diagnostics":plan.diagnostics,"candidates":plan.candidates})
        def fail(exc):
            if current():
                log.warn("Pickup planning failed: %s",exc)
                self._result=ActionResult(status="failure",reason=str(exc),failure=getattr(exc,"failure",FailureInfo()))
        context.run_in_context_loop(work,callback=finish,error_callback=fail)

    def check_result(self, context):
        return self._result or ActionResult(status="running")

    def cancel(self, context):
        self._cancel_requested=True
        self._generation+=1


class PlanHsrPickupRetreat(AsyncAction):
    """Plan retreat and connected undock once the measured-state lift has finished."""
    def __init__(self, plan_key: str):
        super().__init__()
        self.plan_key = plan_key
        self._generation = 0

    def start(self, context):
        self._generation += 1
        generation = self._generation
        self._cancel_requested = False
        self._result = None
        def current():
            return generation == self._generation and not self._cancel_requested and not context.execution.cancel_requested and context.execution.control is None
        previous = context.world_state.get(self.plan_key)
        def work():
            if previous is None:
                raise ValueError("missing pickup plan for post-lift retreat")
            client = PickupPlanningClient(context.ros2_node, PickupBackendSettings.from_hardware(context.hardware))
            plan = client.plan_post_lift(context, previous, cancelled=lambda:not current())
            return client, plan
        def finish(value):
            if not current():
                return
            client, plan = value
            self._client = client
            if not plan.motions:
                fail(RuntimeError(plan.diagnostics["reason"]))
                return
            for motion in plan.motions:
                context.world_state.set(f"{self.plan_key}.{motion.name}", motion.trajectory)
            context.world_state.set(f"{self.plan_key}.retreat_mode", plan.selected_mode)
            log.info("Retreat selected mode=%s (insertion mode=%s)",plan.selected_mode,previous.selected_mode)
            try:
                client.publish(plan)
            except Exception as exc:
                log.warn("Post-lift RViz publication failed (plan retained): %s",exc)
            self._result = ActionResult(status="success")
        def fail(exc):
            if current():
                log.warn("Post-lift retreat planning failed: %s",exc)
                self._result = ActionResult(status="failure",reason=str(exc),failure=getattr(exc,"failure",FailureInfo(FailureCode.PLANNING_FAILED)))
        context.run_in_context_loop(work,callback=finish,error_callback=fail)

    def check_result(self, context):
        return self._result or ActionResult(status="running")

    def cancel(self, context):
        self._cancel_requested = True
        self._generation += 1


class HsrPickup(AbstractAction):
    """Pick using auto/base_only/arm_only/coordinated insertion, then undock and go.

    target_collision_geometry selects regular voxel mesh (default) or convex_hull.
    collision_hull_inflation_factor applies only to convex_hull.

    auto tries complete pickups in the native server's priority. Staging tries base
    first, arm first, then coordinated. Lift uses measured relative joints;
    retreat/undock are replanned after lift from measured state. Go uses the existing named-arm action. Successful cleanup
    matches PickTargetWithGraspAlignment.
    """
    required_controllers=["arm","gripper"]

    def __init__(self,target: str,approach_mode: ApproachMode="auto",
                 backoff_direction: BackoffDirection="grasp_axis",pregrasp_distance_m: float=.25,
                 min_insertion_m: float=.04,max_insertion_m: float=.60,
                 clearance_m: float=.03,post_grasp_lift_m: float=.05,
                 collision_hull_inflation_factor: float=2.,force_N: float=.5,width_m: float=.15,
                 include_surround_collision: bool=True,
                 target_collision_geometry: TargetCollisionGeometry="mesh",
                 failure_undock_mode: Literal['base_backwards','reverse_insertion']='base_backwards',
                 target_label: str | None = None):
        if failure_undock_mode not in ('base_backwards','reverse_insertion'):
            raise ValueError('failure_undock_mode must be base_backwards or reverse_insertion')
        self.failure_undock_mode=failure_undock_mode
        self.target,self.approach_mode=target,approach_mode
        self.target_label=target_label
        self.backoff_direction,self.pregrasp_distance_m=backoff_direction,pregrasp_distance_m
        self.min_insertion_m,self.max_insertion_m=min_insertion_m,max_insertion_m
        self.clearance_m,self.post_grasp_lift_m=clearance_m,post_grasp_lift_m
        self.collision_hull_inflation_factor=collision_hull_inflation_factor
        self.force_N,self.width_m=force_N,width_m
        self.include_surround_collision=include_surround_collision
        self.target_collision_geometry=target_collision_geometry

    def build_node(self,context,name):
        from src.plugin.actions.manipulation import (_AddCollisionMeshAction,_AttachCollisionObjectAction,
            _AllowCollisionPairsAction,_ClearOctomapAction,_ClearCollisionSceneAction,
            MoveArmToState,ReleaseObjectAction,GraspObjectAction)
        from src.plugin.actions.trajectory import ExecuteJointTrajectory
        key=f"{self.target}_pickup_plan"
        self._key=key
        settings=PickupBackendSettings.from_hardware(context.hardware)
        target_id=f"orchid_pickup_{self.target}"
        surround_id=f"{target_id}_surround"
        self._mesh=_AddCollisionMeshAction(name=target_id,convex_hull=self.target_collision_geometry=="convex_hull",
            inflation_factor=self.collision_hull_inflation_factor,required=True)
        self._surround_mesh=_AddCollisionMeshAction(name=surround_id)
        def optional_surround(action,label):
            return py_trees.composites.Selector(name=label,memory=True,children=[
                BtConditionLeaf(lambda ctx:not self._has_surround,context,"No surrounding collision geometry"),
                action.build_node(context,label)])
        planning=PlanHsrPickup(target=self.target,target_label=self.target_label,approach_mode=self.approach_mode,backoff_direction=self.backoff_direction,
            pregrasp_distance_m=self.pregrasp_distance_m,min_insertion_m=self.min_insertion_m,max_insertion_m=self.max_insertion_m,
            clearance_m=self.clearance_m,post_grasp_lift_m=self.post_grasp_lift_m,
            collision_hull_inflation_factor=self.collision_hull_inflation_factor,open_width_m=self.width_m,
            target_collision_geometry=self.target_collision_geometry)
        def condition(label,field,value):
            return BtConditionLeaf(lambda ctx:ctx.world_state.get(f"{key}.{field}")==value,context,label)
        def execute(label,operation,joints=None):
            message="Starting pickup execution" if operation in {"stage_base_first","stage_arm_first","staging_coordinated"} else None
            return ExecuteJointTrajectory(f"{key}.{operation}",list(joints) if joints else None,start_message=message).build_node(context,label)
        staging=py_trees.composites.Selector(name="Selected staging ordering",memory=True,children=[
            py_trees.composites.Sequence(name="Base then arm",memory=True,children=[condition("Base-first selected","staging_order","base_first"),execute("Stage base","stage_base_first",BASE_JOINTS),execute("Stage arm","stage_arm_second",ARM_JOINTS)]),
            py_trees.composites.Sequence(name="Arm then base",memory=True,children=[condition("Arm-first selected","staging_order","arm_first"),execute("Stage arm","stage_arm_first",ARM_JOINTS),execute("Stage base","stage_base_second",BASE_JOINTS)]),
            py_trees.composites.Sequence(name="Coordinated staging",memory=True,children=[condition("Coordinated staging selected","staging_order","coordinated"),execute("Stage arm and base","staging_coordinated")])])
        insertion=py_trees.composites.Selector(name="Selected insertion mode",memory=True,children=[
            py_trees.composites.Sequence(name=mode+" insertion",memory=True,children=[condition(mode+" selected","mode",mode),execute("Insert "+mode,"insertion",joints)])
            for mode,joints in (("base_only",BASE_JOINTS),("arm_only",ARM_JOINTS),("coordinated",None))])
        self._close=GraspObjectAction(force_N=self.force_N,width_m=self.width_m).build_node(context,"Close gripper")
        close_retry=py_trees.decorators.Retry(name="Retry failed close",child=self._close,num_failures=2)
        close=py_trees.decorators.EternalGuard(name="Close retry eligibility",child=close_retry,
            condition=lambda:retry_eligible(context,self._close,close_retry))
        open_leaf=ReleaseObjectAction(distance=self.width_m).build_node(context,"Open gripper")
        open_retry=py_trees.decorators.Retry(name="Retry failed open",child=open_leaf,num_failures=2)
        opening=py_trees.decorators.EternalGuard(name="Open retry eligibility",child=open_retry,
            condition=lambda:retry_eligible(context,open_leaf,open_retry))
        children=[_ClearCollisionSceneAction().build_node(context,"Clear scene before pick"),
                  self._mesh.build_node(context,"Model target")]
        if self.include_surround_collision:
            children.append(optional_surround(self._surround_mesh,"Model surroundings"))
        children.extend([_ClearOctomapAction().build_node(context,"Clear octomap"),
                  planning.build_node(context,"Plan complete pickup"),staging,opening,
                  _AllowCollisionPairsAction([(n,obj) for n in settings.touch_links for obj in (target_id,"<octomap>")],required=True).build_node(context,"Allow insertion contact")])
        if self.include_surround_collision:
            children.append(optional_surround(_AllowCollisionPairsAction([(n,surround_id) for n in settings.touch_links],required=True),"Allow insertion surroundings"))
        children.extend([insertion,close,
                  _AttachCollisionObjectAction(target_id,"hand_palm_link",list(settings.touch_links)).build_node(context,"Attach target")])
        if self.include_surround_collision:
            surround_pairs=[(target_id,surround_id)]+[(n,surround_id) for n in settings.touch_links]
            children.append(optional_surround(_AllowCollisionPairsAction(surround_pairs,required=True),"Allow held object surroundings"))
        post_pairs=[(n,target_id) for n in settings.touch_links]
        # Preserve the same hand/environment policy through post-close motion.
        post_pairs.extend((n,"<octomap>") for n in settings.touch_links)
        if settings.allow_held_target_octomap_contact:
            post_pairs.append((target_id,"<octomap>"))
        children.append(_AllowCollisionPairsAction(post_pairs,required=True).build_node(context,"Apply held target contact policy"))
        if self.post_grasp_lift_m:
            from src.plugin.actions.manipulation import _MoveJointsRelativeAction
            children.append(_MoveJointsRelativeAction({"arm_lift_joint":self.post_grasp_lift_m}).build_node(context,"Lift attached target from measured joints"))
        children.append(PlanHsrPickupRetreat(key).build_node(context,"Plan retreat from measured post-lift state"))
        children.append(py_trees.composites.Selector(name="Selected raised retreat mode",memory=True,children=[
            py_trees.composites.Sequence(name=mode+" raised retreat",memory=True,children=[
                condition(mode+" retreat selected","retreat_mode",mode),execute("Retreat to raised insertion start: "+mode,"retreat",joints)])
            for mode,joints in (("arm_only",ARM_JOINTS),("arm_yaw",None),("coordinated",None))]))
        children.extend([execute("Undock without turning","undock",BASE_JOINTS),
                         MoveArmToState(state="go").build_node(context,"Stow arm to go"),
                         _ClearCollisionSceneAction().build_node(context,"Clear scene after pick"),_ClearOctomapAction().build_node(context,"Clear octomap after pick")])
        self._nodes=children
        operations=py_trees.composites.Sequence(name="HSR pickup operations",memory=True,children=children)
        self._operations=operations
        from src.plugin.actions.navigation import MoveBaseWithWholeBody
        escape=MoveBaseWithWholeBody(frame_id='odom',relative_x=-.30,relative_y=0.,relative_yaw=0.,plan_motion=False,
                                     relative_in_base_frame=self.failure_undock_mode=='base_backwards')
        def bind_escape(ctx):
            failed=terminal_failure(ctx,operations)
            if self.failure_undock_mode=='base_backwards':
                log.warn('Pickup failed after insertion: %s; fallback base_backwards 0.30 m from current pose, hold current yaw, then direct arm go',failed.reason if failed else 'unknown')
                return
            plan=ctx.world_state.get(key)
            if plan is None or plan.inputs.frame_id!='odom':
                return ActionResult(status='failure',reason='fallback undock requires the odom-frame pickup insertion',failure=FailureInfo(FailureCode.INVALID_STATE))
            motion=next((m for m in plan.motions if m.name=='insertion'),None)
            if motion is None or len(motion.hand_path)<2:
                return ActionResult(status='failure',reason='fallback undock missing insertion direction',failure=FailureInfo(FailureCode.INVALID_STATE))
            direction=np.asarray(motion.hand_path[0],dtype=float)[:2]-np.asarray(motion.hand_path[-1],dtype=float)[:2]
            distance=float(np.linalg.norm(direction))
            if not np.isfinite(distance) or distance<1e-6:
                return ActionResult(status='failure',reason='fallback undock insertion has no planar direction',failure=FailureInfo(FailureCode.INVALID_STATE))
            escape.relative_x,escape.relative_y=map(float,.30*direction/distance)
            failed=terminal_failure(ctx,operations)
            log.warn('Pickup failed after insertion: %s; direct fallback undock delta_xy=(%.3f, %.3f) m, hold current yaw, then direct arm go',
                     failed.reason if failed else 'unknown',escape.relative_x,escape.relative_y)
        recovery=ActionScope(self,context,'Post-insertion failure escape',
            py_trees.composites.Sequence(name='Undock then direct arm go',memory=True,children=[
                escape.build_node(context,'Fallback: undock 30 cm '+self.failure_undock_mode+' without turning'),
                MoveArmToState(state='go',force_execution=True).build_node(context,'Fallback: direct arm go')]),
            on_enter=bind_escape,
            on_complete=lambda ctx,status:ActionResult(status='success' if status==py_trees.common.Status.SUCCESS else 'failure'),
            on_cancel=lambda ctx:None)
        optional_recovery=py_trees.composites.Selector(name='Recover failed pickup',memory=True,children=[
            py_trees.composites.Sequence(name='Before insertion: return arm to go',memory=True,children=[
                BtConditionLeaf(lambda ctx:insertion.status==py_trees.common.Status.INVALID,
                                context,'Insertion not started'),
                MoveArmToState(state='go').build_node(context,'Fallback before insertion: arm to go')]),
            py_trees.composites.Sequence(name='After insertion: undock then go',memory=True,children=[
                BtConditionLeaf(lambda ctx:insertion.status==py_trees.common.Status.SUCCESS,context,'Insertion completed'),recovery]),
            py_trees.behaviours.Success(name='Insertion interrupted: skip additional motion')])
        cleanup=py_trees.composites.Sequence(name="Cleanup failed pickup",memory=True,children=[
            py_trees.decorators.FailureIsSuccess(name='Always continue scene cleanup after escape',child=optional_recovery),
            _ClearCollisionSceneAction().build_node(context,"Clear failed pickup scene"),
            _ClearOctomapAction().build_node(context,"Clear failed pickup octomap"),
            py_trees.behaviours.Failure(name="Preserve pickup failure")])
        return ActionScope(self,context,name,py_trees.composites.Selector(name="Pickup with cleanup",memory=True,children=[operations,cleanup]))

    def on_enter(self,context):
        PickupRequest(target=self.target,approach_mode=self.approach_mode,backoff_direction=self.backoff_direction,
            pregrasp_distance_m=self.pregrasp_distance_m,min_insertion_m=self.min_insertion_m,max_insertion_m=self.max_insertion_m,
            clearance_m=self.clearance_m,post_grasp_lift_m=self.post_grasp_lift_m,
            collision_hull_inflation_factor=self.collision_hull_inflation_factor,open_width_m=self.width_m,
            target_collision_geometry=self.target_collision_geometry)
        observation=pickup_observation(context,self.target)
        self._mesh.pcd=observation.target_points
        self._mesh.frame_id=observation.frame_id
        self._surround_mesh.pcd=observation.surround_points
        self._surround_mesh.frame_id=observation.frame_id
        self._has_surround=bool(self.include_surround_collision and observation.surround_points.size)
        if not observation.frame_id or observation.target_points.ndim!=2 or observation.target_points.shape[1]!=3:
            return ActionResult(status="failure",reason="missing target point cloud/frame",failure=FailureInfo(FailureCode.INVALID_STATE))
        for suffix in ("mode","staging_order","stage_base_first","stage_arm_second","stage_arm_first","stage_base_second","staging_coordinated","insertion"):
            context.world_state.set(f"{self._key}.{suffix}",None)

    def on_complete(self,context,status):
        results=[getattr(n,"result",None) for root in self._nodes for n in root.iterate() if n.status==py_trees.common.Status.FAILURE]
        if status==py_trees.common.Status.SUCCESS:
            context.world_state.set(f"{self.target}_held_state","commanded")
            return ActionResult(status="success",payload={"holding":"commanded","target":self.target})
        # Native descendants record their terminal result; preserve the actual cause.
        failed=terminal_failure(context,self._operations)
        if self._close.started:
            holding="commanded" if self._close.result and self._close.result.status=="success" else "uncertain"
            context.world_state.set(f"{self.target}_held_state",holding)
            return ActionResult(status="failure",reason=failed.reason if failed else "pickup failed after closure",failure=failed.failure if failed else FailureInfo(),payload={"holding":holding,"target":self.target})
        return failed or ActionResult(status="failure",reason="pickup failed")

    def on_cancel(self,context):
        if self._close.started:
            context.world_state.set(f"{self.target}_held_state","commanded" if self._close.result and self._close.result.status=="success" else "uncertain")
        context.run_in_context_loop(context.require_arm().clear_collision_scene,
            error_callback=lambda exc:log.warn("Pickup scene cleanup after cancellation failed: %s",exc))
