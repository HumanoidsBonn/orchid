#!/usr/bin/env python3
"""Convert unknown cells in a ROS map PGM to free cells."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _read_pgm(path: Path) -> tuple[bytes, int, int, int, bytearray]:
    raw = path.read_bytes()
    tokens: list[bytes] = []
    positions: list[int] = []
    index = 0
    while len(tokens) < 4:
        while index < len(raw) and raw[index:index + 1].isspace():
            index += 1
        if index < len(raw) and raw[index:index + 1] == b"#":
            while index < len(raw) and raw[index:index + 1] not in {b"\n", b"\r"}:
                index += 1
            continue
        start = index
        while index < len(raw) and not raw[index:index + 1].isspace():
            index += 1
        if start == index:
            raise ValueError(f"Invalid PGM header in {path}")
        tokens.append(raw[start:index])
        positions.append(start)
    while index < len(raw) and raw[index:index + 1].isspace():
        index += 1

    magic = tokens[0]
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    if magic not in {b"P2", b"P5"}:
        raise ValueError(f"Unsupported PGM magic {magic!r}; expected P2 or P5")
    if max_value > 255:
        raise ValueError("Only 8-bit PGM maps are supported")
    if magic == b"P5":
        pixels = bytearray(raw[index:index + width * height])
        if len(pixels) != width * height:
            raise ValueError(f"PGM pixel count mismatch in {path}")
        return magic, width, height, max_value, pixels

    ascii_tokens = raw[index:].split()
    if len(ascii_tokens) < width * height:
        raise ValueError(f"PGM pixel count mismatch in {path}")
    pixels = bytearray(int(token) for token in ascii_tokens[: width * height])
    return magic, width, height, max_value, pixels


def _write_pgm(path: Path, magic: bytes, width: int, height: int, max_value: int, pixels: bytearray) -> None:
    if magic == b"P2":
        lines = [b"P2", f"{width} {height}".encode(), str(max_value).encode()]
        for row in range(height):
            start = row * width
            lines.append(b" ".join(str(value).encode() for value in pixels[start:start + width]))
        path.write_bytes(b"\n".join(lines) + b"\n")
        return
    path.write_bytes(b"P5\n" + f"{width} {height}\n{max_value}\n".encode() + bytes(pixels))


def _load_yaml(path: Path) -> dict:
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Map YAML must contain a mapping: {path}")
    return data


def _dump_yaml(path: Path, data: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _resolve_inputs(input_path: Path, output_path: Path | None) -> tuple[Path, Path, Path | None, Path | None, dict | None]:
    if input_path.suffix.lower() in {".yaml", ".yml"}:
        yaml_data = _load_yaml(input_path)
        image = yaml_data.get("image")
        if not image:
            raise ValueError(f"Map YAML has no image field: {input_path}")
        pgm_input = (input_path.parent / str(image)).resolve()
        if output_path and output_path.suffix.lower() == ".pgm":
            pgm_output = output_path
            yaml_output = output_path.with_suffix(input_path.suffix)
        else:
            yaml_output = output_path or input_path.with_name(f"{input_path.stem}_unknown_free{input_path.suffix}")
            pgm_output = yaml_output.with_suffix(".pgm")
        return pgm_input, pgm_output, input_path, yaml_output, yaml_data

    pgm_input = input_path
    pgm_output = output_path or input_path.with_name(f"{input_path.stem}_unknown_free{input_path.suffix}")
    return pgm_input, pgm_output, None, None, None


def convert_unknown_to_free(
    input_path: Path,
    *,
    output_path: Path | None,
    unknown_value: int,
    free_value: int,
) -> dict:
    pgm_input, pgm_output, yaml_input, yaml_output, yaml_data = _resolve_inputs(input_path.resolve(), output_path)
    magic, width, height, max_value, pixels = _read_pgm(pgm_input)
    replaced = 0
    for index, value in enumerate(pixels):
        if value == unknown_value:
            pixels[index] = free_value
            replaced += 1
    pgm_output.parent.mkdir(parents=True, exist_ok=True)
    _write_pgm(pgm_output, magic, width, height, max_value, pixels)

    if yaml_data is not None and yaml_output is not None:
        yaml_output.parent.mkdir(parents=True, exist_ok=True)
        yaml_data = dict(yaml_data)
        yaml_data["image"] = pgm_output.name
        _dump_yaml(yaml_output, yaml_data)
    elif yaml_input is not None and yaml_output is not None:
        shutil.copy2(yaml_input, yaml_output)

    return {
        "input": str(input_path),
        "pgm_input": str(pgm_input),
        "pgm_output": str(pgm_output),
        "yaml_output": str(yaml_output) if yaml_output else None,
        "width": width,
        "height": height,
        "unknown_value": unknown_value,
        "free_value": free_value,
        "replaced": replaced,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input ROS map YAML or PGM file.")
    parser.add_argument("-o", "--output", help="Output YAML or PGM path. Defaults to *_unknown_free.")
    parser.add_argument("--unknown-value", type=int, default=205)
    parser.add_argument("--free-value", type=int, default=254)
    args = parser.parse_args()

    result = convert_unknown_to_free(
        Path(args.input),
        output_path=Path(args.output).resolve() if args.output else None,
        unknown_value=int(args.unknown_value),
        free_value=int(args.free_value),
    )
    print(
        "Converted {replaced} unknown cells in {pgm_input} -> {pgm_output}".format(
            **result
        )
    )
    if result["yaml_output"]:
        print(f"Wrote map YAML {result['yaml_output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
