"""Synthetic preview geometry and publication tests; no perception or motion."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest

from src.core.context import Context
from src.core.manipulation_types import BasePose2D
from src.plugin.actions.perception_preview import (
    ObservationImage,capped_grasp_width,segmentation_image,observation_preview,
    grasp_image,gripper_points,publish_grasp_preview,publish_image,
)


def test_segmentation_is_nonblocking_and_preserves_source():
    source=np.zeros((80,100,3),dtype=np.uint8)
    result=segmentation_image(source,[20,20,60,70],[[20,20,70,20,70,60,20,60]],'mug')
    assert not source.any()
    assert result[40,40,1]>0 and result.shape==source.shape


@pytest.mark.parametrize('width',[.04,.15,.25])
def test_grasp_width_cap_and_physical_opening(width):
    metadata=capped_grasp_width({'estimated_width_m':width})
    assert metadata['raw_estimated_width_m']==width
    assert metadata['estimated_width_m']==min(width,.15)
    assert capped_grasp_width(metadata)==metadata
    points=gripper_points(metadata['estimated_width_m'])
    assert points[4,1]-points[1,1]==pytest.approx(min(width,.15))


@pytest.mark.parametrize('mode',['cylinder_side','short_axis_side','top_down'])
def test_estimator_metadata_caps_width_without_changing_pose(mode):
    from src.plugin.actions.grasp_alignment import simple_pca_grasp_pose
    points=np.array([[x,y,z] for x in (-.2,.2) for y in (-.12,.12) for z in (.7,.9)])
    pose,metadata=simple_pca_grasp_pose(points,frame_id='odom',robot_pose=BasePose2D(-1,0,0),grasp_mode=mode)
    assert metadata['raw_estimated_width_m']>=.15
    assert metadata['estimated_width_m']==.15
    assert pose.shape==(4,4) and np.isfinite(pose).all()


def test_observation_snapshot_scales_calibration_and_uses_image_timestamp(monkeypatch):
    context=Context(sensor_data_center=Mock())
    context.sensor_data_center.get_pose_at.return_value={'position':{'x':1.,'y':0.,'z':0.},'orientation':{'x':0.,'y':0.,'z':0.,'w':1.}}
    header=NS(frame_id='optical',stamp=NS(sec=12,nanosec=500000000))
    info=NS(width=640,height=480,p=[400.,0,320.,0,0,400.,240.,0,0,0,1.,0],k=[0.]*9,d=[],header=header)
    action=NS(_context=context,target='mug',processed_frame=np.zeros((240,320,3),np.uint8),rgb_data=NS(data=NS(header=header)),camera_info_data=NS(data=info))
    published=Mock();monkeypatch.setattr('src.plugin.actions.perception_preview.publish_image',published)
    observation_preview(action,[20,20,60,70],[],'odom')
    snapshot=context.world_state.get('perception_preview_observation')
    assert snapshot.camera_matrix[0,0]==200. and snapshot.camera_matrix[0,2]==160.
    assert snapshot.camera_from_world[0,3]==-1.
    context.sensor_data_center.get_pose_at.assert_called_once_with('optical','odom',timestamp=12.5)
    published.assert_called_once()
    action.camera_info_data=None
    observation_preview(action,[20,20,60,70],[],'odom')
    assert context.world_state.get('perception_preview_observation') is None
    assert published.call_count==2  # Segmentation still works without calibrated grasp projection.


def test_gripper_projection_and_missing_source_are_optional():
    image=np.zeros((240,320,3),np.uint8)
    source=ObservationImage('mug',image,np.array([[200.,0,160],[0,200.,120],[0,0,1]]),np.zeros(5),np.eye(4),'odom',None)
    pose=np.eye(4);pose[2,3]=1.
    assert grasp_image(source,pose,.08).any()
    assert not source.image.any()
    pose[2,3]=-1.
    with pytest.raises(ValueError,match='behind'):grasp_image(source,pose,.08)
    context=Context()
    publish_grasp_preview(context,'mug',pose,'odom','+z',.08)
    assert not context.perception_preview_publishers


def test_publisher_is_robot_scoped_bounded_and_reused():
    from std_msgs.msg import Header
    context=Context(robot_id='robot',ros2_node=Mock())
    image=np.zeros((30,40,3),np.uint8)
    publish_image(context,'segmentation',image,Header())
    publish_image(context,'segmentation',image,Header())
    context.ros2_node.create_publisher.assert_called_once()
    args=context.ros2_node.create_publisher.call_args.args
    assert args[1]=='/robot/visualization/segmentation/compressed' and args[2].depth==1
    assert context.ros2_node.create_publisher.return_value.publish.call_count==2


@pytest.mark.parametrize('name',['EstimateSimplePcaGrasp','EstimateSimpleCylindricalGrasp'])
def test_estimator_succeeds_without_preview_source_and_caps_world_state(monkeypatch,name):
    from src.plugin.actions import grasp_alignment
    context=Context()
    points=np.array([[x,y,z] for x in (-.2,.2) for y in (-.12,.12) for z in (.7,.9)])
    context.world_state.set('mug_object_pcd',points)
    context.world_state.set('mug_pcd_frame','odom')
    monkeypatch.setattr(grasp_alignment,'_current_base_pose',lambda *a:BasePose2D(-1,0,0))
    action=getattr(grasp_alignment,name)(target='mug',debug_visualization=False)
    result=action.execute(context)
    assert result.status=='success'
    assert context.world_state.get('mug_estimated_width')==.15
    assert result.payload['metadata']['estimated_width_m']==.15
    assert result.payload['metadata']['raw_estimated_width_m']>.15
