"""Read-only checks for the maintained scene files and catalog integration."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from src.core.scene_graph import SceneGraph
from src.core.scene_graph_query import SceneGraphQuery
from src.orchestrator.session import RobotSession
from src.workflows.catalog import WorkflowCatalog, WorkflowSettings, DEFINITIONS, visitable_locations

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / 'config/scene/scene_graph.json'
UI = ROOT / 'submodules/orchid_ui/static/scene_graph_complete.json'


def test_copies_match_and_have_no_duplicate_ids_or_dangling_support():
    assert SERVER.read_bytes() == UI.read_bytes()
    assert (SERVER.parent/'support_relationships.yaml').read_bytes() == (UI.parent/'support_relationships.yaml').read_bytes()
    data = json.loads(SERVER.read_text())
    assert data['metadata']['relationship_config'] == 'support_relationships.yaml'
    ids = [o['id'] for o in data['objects']]
    assert len(ids) == len(set(ids))
    assert all(o.get('supported_by') is None or o['supported_by'] in ids for o in data['objects'])
    assert data['metadata']['unique_objects'] == len(ids)
    assert data['metadata']['supported_by_relationships'] == sum(o.get('supported_by') is not None for o in data['objects'])


def test_default_server_load_and_mug_inherits_desk_access_poses(monkeypatch):
    monkeypatch.delenv('ORCHID_SCENE_GRAPH_PATH', raising=False)
    graph = RobotSession._load_scene_graph()
    assert graph is not None
    query = SceneGraphQuery(graph)
    mug = next(n for n in graph.get_nodes('object') if n.attributes['label'] == 'coffee_mug')
    assert mug.attributes['manually_added'] is True
    assert mug.attributes['center'] == [-8.009886, -3.403507, .784774]
    assert mug.attributes['size'] == [.08, .08, .10]
    assert query.get_supporting_furniture(mug.id) == 'work_desk_41'
    assert query.get_visit_locations(mug.id) == query.get_visit_locations('work_desk_41')
    assert len(query.get_visit_locations(mug.id)) == 4
    assert len(visitable_locations(SimpleNamespace(scene_graph=graph))) == 19


def test_scene_choices_keep_extras_and_deduplicate_space_underscore_labels():
    graph = SceneGraph.from_json(str(SERVER))
    settings = WorkflowSettings(items=[{'id':'pringles','label':'Pringles can','query':'pringles can'},
                                     {'id':'absent','label':'Absent item','query':'absent item'}])
    choices = WorkflowCatalog(SimpleNamespace(scene_graph=graph), 'test', settings).items()
    assert any(x['query'] == 'coffee_mug' for x in choices)
    assert any(x['id'] == 'absent' for x in choices)
    assert sum(x['query'].replace('_',' ').casefold() == 'pringles can' for x in choices) == 1
    assert not any(x['query'] in {'work_desk','wall','window'} for x in choices)


def test_explicit_scene_path_resolves_neighbour_yaml(monkeypatch, tmp_path):
    path = tmp_path/'selected.json'
    path.write_bytes(SERVER.read_bytes())
    (tmp_path/'support_relationships.yaml').write_bytes((SERVER.parent/'support_relationships.yaml').read_bytes())
    monkeypatch.setenv('ORCHID_SCENE_GRAPH_PATH', str(path))
    assert RobotSession._load_scene_graph().get_node('work_desk_41') is not None


@pytest.mark.parametrize('missing_yaml', [False, True])
def test_missing_data_reports_active_path_and_returns_no_graph(monkeypatch, tmp_path, missing_yaml):
    path = tmp_path/'selected.json'
    if missing_yaml:
        path.write_bytes(SERVER.read_bytes())
    monkeypatch.setenv('ORCHID_SCENE_GRAPH_PATH', str(path))
    warning = Mock()
    monkeypatch.setattr('src.orchestrator.session.log.warn', warning)
    assert RobotSession._load_scene_graph() is None
    assert warning.call_count == 1
    assert str(path) in str(warning.call_args)
    assert ('support_relationships.yaml' if missing_yaml else 'selected.json') in str(warning.call_args)


def test_missing_graph_and_no_visit_poses_are_distinct_catalog_states():
    definition = DEFINITIONS['fetch']
    ctx = SimpleNamespace(scene_graph=None, supported_actions={k:object for k in definition.required_actions},
                          nav=object(), arm=object(), head=object(), gripper=object())
    catalog = WorkflowCatalog(ctx,'test',WorkflowSettings())
    assert catalog.unavailable_reason(definition) == 'Missing: scene graph not loaded'
    ctx.scene_graph = SceneGraph()
    assert catalog.unavailable_reason(definition) == 'Missing: valid furniture access poses in loaded scene graph'


def test_disabling_desk_poses_keeps_objects_but_removes_visit_location():
    data = json.loads(SERVER.read_text())
    desk = next(o for o in data['objects'] if o['id'] == 41)
    for pose in desk['base_access_poses']:
        pose['valid'] = False
    graph = SceneGraph.from_dict(data, SERVER.parent/'support_relationships.yaml')
    assert graph.get_node('work_desk_41') is not None
    mug = next(n for n in graph.get_nodes('object') if n.attributes['label'] == 'coffee_mug')
    assert SceneGraphQuery(graph).get_best_visit_location(mug.id) is None
    assert 'work_desk_41' not in {row['id'] for row in visitable_locations(SimpleNamespace(scene_graph=graph))}
