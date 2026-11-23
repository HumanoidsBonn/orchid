"""Constructor validation shared by registered action and condition inputs."""
import inspect
from typing import Any, get_type_hints
from pydantic import TypeAdapter


def instantiate_inputs(cls: type, params: dict[str, Any]):
    signature = inspect.signature(cls)
    declared = {name for name, field in signature.parameters.items()
                if field.kind not in (field.VAR_POSITIONAL, field.VAR_KEYWORD)}
    unknown = params.keys() - declared
    if unknown:
        raise ValueError(f"Unknown inputs for {cls.__name__}: {', '.join(sorted(unknown))}")
    signature.bind(**params)
    annotations = get_type_hints(cls.__init__)
    validated = {name: TypeAdapter(annotations.get(name, Any)).validate_python(value)
                 for name, value in params.items()}
    return cls(**validated)
