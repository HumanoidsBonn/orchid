from unittest.mock import Mock

import numpy as np
import pytest

from src.core.context import Context
from src.plugin.actions.navigation import AlignViewToSceneTarget, MoveToTarget


@pytest.mark.parametrize('head,lift',[(True,True),(False,True),(True,False)])
def test_alignment_goals_only_contain_enabled_joints(head,lift):
    action=AlignViewToSceneTarget(align_head=head,align_lift=lift)
    current={'arm_lift_joint':.23,'head_pan_joint':.17,'head_tilt_joint':-.31}
    limits={'arm_lift_joint':(0.,.69),'head_pan_joint':(-3.,3.),'head_tilt_joint':(-1.5,.5)}
    rows=action._candidate_view_goals(np.array([[1.,.3,1.1],[1.2,.4,1.1]]),current,limits,None)
    expected=({'arm_lift_joint'} if lift else set()) | ({'head_pan_joint','head_tilt_joint'} if head else set())
    assert rows and all(set(row['goal'])==expected for row in rows)
    if not lift:
        assert all(row['camera_z']==pytest.approx(action.camera_height_m+current['arm_lift_joint']) for row in rows)


def test_both_alignment_options_disabled_skip_without_scene_or_controller():
    assert AlignViewToSceneTarget(align_head=False,align_lift=False)._solve(Context())['goal']=={}


def test_move_target_keeps_lift_alignment_without_post_navigation_head_alignment():
    action=MoveToTarget(target='table')
    action.build_node(Context(nav=Mock(),arm=Mock(),head=Mock()),'Move')
    assert action._align.align_head is False and action._align.align_lift is True
