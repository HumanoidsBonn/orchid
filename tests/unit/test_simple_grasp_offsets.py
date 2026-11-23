"""Pure offset geometry and input checks; no perception/action execution."""
import math
import numpy as np
import pytest
from src.core.manipulation_types import BasePose2D
from src.plugin.actions.grasp_alignment import (
    EstimateSimplePcaGrasp, EstimateSimpleCylindricalGrasp,
    simple_pca_grasp_pose, simple_cylindrical_grasp_pose,
)
from src.plugin.actions.hsr_pickup.planning import import_simple_grasp
from src.plugin.actions.hsr_pickup.model import PickupRequest,pickup_geometry


def cloud(angle=0., round_shape=False, long_half=.06):
    long=np.array([math.cos(angle),math.sin(angle),0.])
    short=np.array([-math.sin(angle),math.cos(angle),0.])
    centre=np.array([1.,.3,.7])
    points=np.array([centre+x*long+y*short+[0.,0.,z]
                     for x in np.linspace(-long_half,long_half,5)
                     for y in np.linspace(-(.06 if round_shape else .02),.06 if round_shape else .02,5)
                     for z in (-.04,.04)])
    return points,centre,long


@pytest.mark.parametrize('axis',['+z','-z','+x','-x'])
@pytest.mark.parametrize('yaw',[0.,.7])
def test_cylinder_offset_follows_signed_approach_toward_robot(axis,yaw):
    points,centre,_=cloud()
    robot=BasePose2D(centre[0]-math.cos(yaw),centre[1]-math.sin(yaw),0.,'odom')
    baseline,old=simple_cylindrical_grasp_pose(points,frame_id='odom',robot_pose=robot,
                                             approach_axis=axis,outward_offset_m=0.)
    pose,meta=simple_cylindrical_grasp_pose(points,frame_id='odom',robot_pose=robot,approach_axis=axis)
    np.testing.assert_allclose(pose[:3,3],centre-.05*np.array([math.cos(yaw),math.sin(yaw),0.]))
    np.testing.assert_array_equal(pose[:3,:3],baseline[:3,:3])
    np.testing.assert_allclose(meta['centroid'],old['centroid'])
    assert meta['outward_offset_m']==.05
    np.testing.assert_allclose(meta['unoffset_grasp_position'],baseline[:3,3])


@pytest.mark.parametrize('angle',[0.,.6,1.4])
@pytest.mark.parametrize('side',[-1,1])
def test_short_axis_uses_its_own_offset_without_changing_opening(angle,side):
    points,centre,long=cloud(angle,long_half=.04)  # Ratio 2 keeps the short-axis strategy.
    xy=centre[:2]+side*long[:2]
    robot=BasePose2D(*xy,0.,'odom')
    baseline,old=simple_pca_grasp_pose(points,frame_id='odom',robot_pose=robot,grasp_mode='short_axis_side',
        cylinder_grasp_offset_m=.3,short_axis_grasp_offset_m=0.)
    pose,meta=simple_pca_grasp_pose(points,frame_id='odom',robot_pose=robot,grasp_mode='short_axis_side',
        cylinder_grasp_offset_m=.3,short_axis_grasp_offset_m=.08)
    np.testing.assert_allclose(pose[:3,3],baseline[:3,3]+side*.08*long)
    np.testing.assert_array_equal(pose[:3,:3],baseline[:3,:3])
    assert meta['estimated_width_m']==old['estimated_width_m']
    palm,_=import_simple_grasp(pose,'+z')
    np.testing.assert_array_equal(palm[:3,3],pose[:3,3])


@pytest.mark.parametrize('round_shape',[False,True])
def test_auto_cylinder_branches_use_cylinder_offset(round_shape):
    points,centre,_=cloud(round_shape=round_shape)
    pose,meta=simple_pca_grasp_pose(points,frame_id='odom',robot_pose=BasePose2D(0.,.3,0.,'odom'),
        grasp_mode='auto',cylinder_grasp_offset_m=.07,short_axis_grasp_offset_m=.2)
    assert meta['selected_strategy']=='cylinder_side'
    np.testing.assert_allclose(pose[:3,3],centre-[.07,0.,0.])


def test_top_down_is_unshifted_and_pregrasp_uses_shifted_endpoint_once():
    points,_,_=cloud()
    args=dict(frame_id='odom',robot_pose=BasePose2D(0.,.3,0.,'odom'))
    a,_=simple_pca_grasp_pose(points,**args,grasp_mode='top_down',cylinder_grasp_offset_m=0.,short_axis_grasp_offset_m=0.)
    b,_=simple_pca_grasp_pose(points,**args,grasp_mode='top_down',cylinder_grasp_offset_m=.2,short_axis_grasp_offset_m=.3)
    np.testing.assert_array_equal(a,b)
    grasp,_=simple_pca_grasp_pose(points,**args,grasp_mode='cylinder_side')
    grasp,_=import_simple_grasp(grasp,'+z')
    geometry=pickup_geometry(PickupRequest('jar'),grasp,points,'odom')
    np.testing.assert_allclose(geometry.grasp,grasp)
    np.testing.assert_allclose(geometry.pregrasp(.25)[:3,3],grasp[:3,3]-.25*grasp[:3,2])


def test_explicit_action_inputs_have_independent_defaults_and_serialization():
    a=EstimateSimplePcaGrasp.from_params(target='jar',cylinder_grasp_offset_m=.02,short_axis_grasp_offset_m=.08)
    assert a.param_dict()['cylinder_grasp_offset_m']==.02
    assert a.param_dict()['short_axis_grasp_offset_m']==.08
    a=EstimateSimplePcaGrasp('jar')
    assert a.cylinder_grasp_offset_m==a.short_axis_grasp_offset_m==.05
    assert EstimateSimpleCylindricalGrasp('jar').param_dict()['cylinder_grasp_offset_m']==.05


@pytest.mark.parametrize('name',['cylinder_grasp_offset_m','short_axis_grasp_offset_m'])
@pytest.mark.parametrize('value',[-.01,float('nan'),float('inf')])
def test_invalid_offsets_rejected(name,value):
    with pytest.raises(ValueError,match='finite and nonnegative'):
        EstimateSimplePcaGrasp('jar',**{name:value})
