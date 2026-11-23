"""Pure strategy-selection checks; no perception, actions or robot execution."""
import numpy as np
import pytest
from src.core.manipulation_types import BasePose2D
from src.plugin.actions import grasp_alignment as grasp


@pytest.mark.parametrize('mode', ['auto', 'short_axis_side'])
@pytest.mark.parametrize('height', [.3, .8])
@pytest.mark.parametrize('ratio,expected', [
    (1., 'cylinder_side'), (1.25, 'cylinder_side'),
    (1.2501, 'short_axis_side'), (2., 'short_axis_side'), (2.5, 'short_axis_side'),
    (2.5001, 'cylinder_side'), (4., 'cylinder_side')])
def test_auto_strategy_ratio_boundaries_are_independent_of_height(monkeypatch, mode, height, ratio, expected):
    points = np.array([[1+x, y, height+z] for x in (-.04, .04)
                       for y in (-.02, .02) for z in (-.02, .02)])
    analysis = grasp._pca_grasp_analysis(points, robot_pose=BasePose2D(0, 0, 0))
    analysis['horizontal_extent_ratio'] = ratio
    monkeypatch.setattr(grasp, '_pca_grasp_analysis', lambda *args, **kwargs: analysis)
    _, metadata = grasp.simple_pca_grasp_pose(points, frame_id='odom',
                                             robot_pose=BasePose2D(0, 0, 0), grasp_mode=mode)
    assert metadata['selected_strategy'] == expected
    assert metadata['grasp_mode'] == mode


@pytest.mark.parametrize('mode', ['cylinder_side', 'top_down'])
@pytest.mark.parametrize('ratio', [1., 2., 4.])
def test_explicit_strategy_overrides_auto_ratio_rules(monkeypatch, mode, ratio):
    points = np.array([[1+x, y, .3+z] for x in (-.04, .04)
                       for y in (-.02, .02) for z in (-.02, .02)])
    analysis = grasp._pca_grasp_analysis(points, robot_pose=BasePose2D(0, 0, 0))
    analysis['horizontal_extent_ratio'] = ratio
    monkeypatch.setattr(grasp, '_pca_grasp_analysis', lambda *args, **kwargs: analysis)
    _, metadata = grasp.simple_pca_grasp_pose(points, frame_id='odom',
                                             robot_pose=BasePose2D(0, 0, 0), grasp_mode=mode)
    assert metadata['selected_strategy'] == mode
