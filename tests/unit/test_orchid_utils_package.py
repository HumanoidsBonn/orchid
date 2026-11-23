from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_twist_mux_config_routes_nav_and_teleop_to_base():
    config_path = REPO_ROOT / "submodules" / "orchid_utils" / "config" / "twist_mux.yaml"
    data = yaml.safe_load(config_path.read_text())

    params = data["twist_mux"]["ros__parameters"]
    topics = params["topics"]
    assert topics["nav"]["topic"] == "/cmd_vel_nav"
    assert topics["teleop"]["topic"] == "/cmd_vel"
    assert topics["teleop"]["priority"] > topics["nav"]["priority"]

    launch_text = (
        REPO_ROOT / "submodules" / "orchid_utils" / "launch" / "twist_mux.launch.py"
    ).read_text()
    assert 'default_value="/omni_base_controller/cmd_vel"' in launch_text
    assert 'remappings=[("cmd_vel_out", output_cmd_vel)]' in launch_text


def test_keyboard_teleop_defaults_are_fast_enough_for_sim():
    launch_text = (
        REPO_ROOT / "submodules" / "orchid_utils" / "launch" / "keyboard_teleop.launch.py"
    ).read_text()
    wrapper_text = (
        REPO_ROOT
        / "submodules"
        / "orchid_utils"
        / "orchid_utils"
        / "turtlebot3_teleop.py"
    ).read_text()

    assert 'DeclareLaunchArgument("max_linear_speed", default_value="2.0")' in launch_text
    assert 'DeclareLaunchArgument("linear_step_size", default_value="0.1")' in launch_text
    assert 'executable="orchid_turtlebot3_teleop"' in launch_text
    assert 'remappings=[("cmd_vel", LaunchConfiguration("cmd_vel"))]' in launch_text
    assert '"ORCHID_TELEOP_MAX_LIN_VEL": LaunchConfiguration("max_linear_speed")' in launch_text
    assert '"ORCHID_TELEOP_LIN_STEP": LaunchConfiguration("linear_step_size")' in launch_text

    assert "from turtlebot3_teleop.script import teleop_keyboard" in wrapper_text
    assert 'teleop_keyboard.BURGER_MAX_LIN_VEL = max_linear' in wrapper_text
    assert 'teleop_keyboard.LIN_VEL_STEP_SIZE = linear_step' in wrapper_text


def test_nav2_cmd_vel_is_remapped_to_mux_input():
    launch_text = (
        REPO_ROOT / "submodules" / "nav2_config" / "launch" / "navigation.launch.py"
    ).read_text()

    assert "velocity_remappings = [('cmd_vel', '/cmd_vel_nav')]" in launch_text
    assert "velocity_remappings = [('cmd_vel', 'omni_base_controller/cmd_vel')]" not in launch_text


def test_terminator_runner_uses_repo_owned_layout():
    runner_text = (REPO_ROOT / "scripts" / "launch_orchid_stack_terminator.sh").read_text()
    config_text = (
        REPO_ROOT / "submodules" / "orchid_utils" / "terminator" / "orchid_stack.config"
    ).read_text()

    assert "--layout orchid_stack" in runner_text
    assert "[[orchid_stack]]" in config_text
    assert "ORCHID_MOVEIT_CMD" in config_text
    assert "ORCHID_TWIST_MUX_CMD" in config_text
    assert "ORCHID_RVIZ_CMD" in config_text
