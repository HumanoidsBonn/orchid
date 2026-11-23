"""Launch construction only: never execute a ROS node or send a goal."""
import importlib.util
from pathlib import Path
import struct
import yaml
import pytest
from launch import LaunchContext
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('orchid_navigation_launch',ROOT/'submodules/nav2_config/launch/navigation.launch.py')
NAV=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NAV)


def setup(tmp_path,**overrides):
    loc=tmp_path/'localization.yaml';nav=tmp_path/'navigation.yaml'
    loc.write_text('image: localization.png\nresolution: 0.05\norigin: [0,0,0]\n')
    nav.write_text('image: navigation.png\nresolution: 0.05\norigin: [0,0,0]\n')
    context=LaunchContext()
    for argument in NAV.declare_arguments():
        context.launch_configurations[argument.name]=perform_substitutions(context,argument.default_value)
    context.launch_configurations.update(localization_map=str(loc),navigation_map=str(nav),params_file=str(ROOT/'submodules/nav2_config/config/nav2_params.yaml'))
    context.launch_configurations.update(overrides)
    nodes={node._Node__node_name if isinstance(node._Node__node_name,str) else perform_substitutions(context,node._Node__node_name):node for node in NAV._launch_setup(context) if isinstance(node,Node)}
    return context,nodes,loc,nav


def parameters(context,node):
    result={}
    for value in evaluate_parameters(context,node._Node__parameters):
        if isinstance(value,dict):result.update(value)
        else:
            data=yaml.safe_load(Path(value).read_text())
            result['_file']=data
    return result


def test_distinct_map_servers_and_static_layers(tmp_path):
    context,nodes,loc,nav=setup(tmp_path)
    a=parameters(context,nodes['localization_map_server']);b=parameters(context,nodes['navigation_map_server'])
    assert a['yaml_filename']==str(loc) and a['topic_name']=='/map_localization'
    assert b['yaml_filename']==str(nav) and b['topic_name']=='/map_navigation'
    assert a['frame_id']==b['frame_id']=='map'
    assert parameters(context,nodes['amcl'])['map_topic']=='/map_localization'
    config=parameters(context,nodes['planner_server'])['_file']
    for kind in ['global_costmap','local_costmap']:
        assert config[kind][kind]['ros__parameters']['static_layer']['map_topic']=='/map_navigation'
    assert list(parameters(context,nodes['lifecycle_manager_localization'])['node_names'])==['localization_map_server','amcl']
    assert 'navigation_map_server' in parameters(context,nodes['lifecycle_manager_navigation'])['node_names']


def test_namespace_and_external_localization(tmp_path):
    context,nodes,_,_=setup(tmp_path,namespace='robot',use_localization='false')
    assert 'amcl' not in nodes and 'localization_map_server' not in nodes
    assert parameters(context,nodes['navigation_map_server'])['topic_name']=='/robot/map_navigation'
    config=parameters(context,nodes['planner_server'])['_file']['robot']
    assert config['global_costmap']['global_costmap']['ros__parameters']['static_layer']['map_topic']=='/robot/map_navigation'


def test_legacy_map_shorthand_and_external_map_compatibility(tmp_path):
    shared=tmp_path/'shared.yaml';shared.write_text('image: shared.png\n')
    context,nodes,_,_=setup(tmp_path,localization_map='',navigation_map='',map=str(shared))
    for name in ['localization_map_server','navigation_map_server']:
        assert parameters(context,nodes[name])['yaml_filename']==str(shared)
    context,nodes,_,_=setup(tmp_path,use_localization='false',localization_map='',navigation_map='',map='')
    assert 'navigation_map_server' not in nodes and 'amcl' not in nodes
    assert parameters(context,nodes['planner_server'])['_file']['global_costmap']['global_costmap']['ros__parameters']['static_layer']['map_topic']=='/map'


def test_missing_map_path_fails_before_launch(tmp_path):
    with pytest.raises(RuntimeError,match='navigation_map YAML does not exist'):
        setup(tmp_path,navigation_map=str(tmp_path/'missing.yaml'))


def test_static_png_registration_matches_its_demo_source():
    static=ROOT/'submodules/orchid_ui/static'
    actual=yaml.safe_load((static/'map.yaml').read_text())
    source=yaml.safe_load((ROOT/'submodules/mapping/maps/demo/map.yaml').read_text())
    assert actual['origin']==source['origin']
    assert actual['resolution']==source['resolution']
    assert actual['image']=='map.png'
    width,height=struct.unpack('>II',(static/'map.png').read_bytes()[16:24])
    assert (width,height)==(626,1056)
