"""Tests for the mapping unknown-to-free utility."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("submodules/mapping/scripts/unknown_to_free.py")
    spec = importlib.util.spec_from_file_location("unknown_to_free", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unknown_to_free_converts_raw_pgm(tmp_path):
    module = _load_module()
    source = tmp_path / "map.pgm"
    target = tmp_path / "map_free.pgm"
    source.write_bytes(b"P2\n3 1\n255\n0 205 254\n")

    result = module.convert_unknown_to_free(source, output_path=target, unknown_value=205, free_value=254)

    assert result["replaced"] == 1
    assert b"0 254 254" in target.read_bytes()


def test_unknown_to_free_converts_yaml_and_updates_image(tmp_path):
    module = _load_module()
    pgm = tmp_path / "map.pgm"
    yaml_file = tmp_path / "map.yaml"
    out_yaml = tmp_path / "map_free.yaml"
    pgm.write_bytes(b"P5\n2 1\n255\n" + bytes([205, 0]))
    yaml_file.write_text("image: map.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n")

    result = module.convert_unknown_to_free(yaml_file, output_path=out_yaml, unknown_value=205, free_value=254)

    out_pgm = tmp_path / "map_free.pgm"
    assert result["replaced"] == 1
    assert out_pgm.read_bytes().endswith(bytes([254, 0]))
    assert "image: map_free.pgm" in out_yaml.read_text()
