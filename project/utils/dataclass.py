"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import json
import pathlib
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple, Union, get_args, get_origin, get_type_hints

from diffusers.utils import BaseOutput


@dataclass
class Dataclass(BaseOutput):
    """
    Base dataclass that
        - turns *every* dict at *any* depth into the matching subtype
        - remains a dict at all levels (`cfg[key]`)
        - offers in-place, recursive, loss-less update()
    """

    __defaults__: Any = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Construction / single-shot hydration
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        super().__post_init__()
        self._already_hydrated = False

        # expose all declared dataclass fields via dict interface
        for f in fields(self):
            if f.name not in self:
                self[f.name] = getattr(self, f.name)

        self._hydrate_once()

    def _hydrate_once(self) -> None:
        """Recursively hydrate the entire tree once per instance."""
        if not self._already_hydrated:
            self._already_hydrated = True
            self._hydrate_recursive()

    def _hydrate_recursive(self) -> None:
        hints = get_type_hints(type(self))
        for f in fields(self):
            raw = getattr(self, f.name)
            new_value = self._deep_convert(raw, hints.get(f.name))

            if new_value is not raw:
                setattr(self, f.name, new_value)
                self[f.name] = new_value

            if isinstance(new_value, Dataclass):
                new_value._hydrate_once()  # cascade — only once

    # ------------------------------------------------------------------
    # Deep converter ----------------------------------------------------
    # ------------------------------------------------------------------
    def _deep_convert(self, raw: Any, target_type: Any) -> Any:
        """Convert raw value to target_type (handles nesting)."""
        if target_type is None:
            return raw

        origin, args = get_origin(target_type), get_args(target_type)

        # Optional[T]
        if origin is Union:
            for arg in args:
                if arg is not type(None):
                    return self._deep_convert(raw, arg)
            return raw

        # List / Tuple / Sequence
        if origin in (list, tuple, Sequence):
            if origin is list and isinstance(raw, list):
                inner_t = args[0] if args else Any
                return [self._deep_convert(item, inner_t) for item in raw]
            if origin is tuple and isinstance(raw, tuple):
                inner_t = args[0] if args else Any
                return tuple(self._deep_convert(item, inner_t) for item in raw)
            if origin is Sequence and isinstance(raw, (list, tuple)):
                inner_t = args[0] if args else Any
                converted = [self._deep_convert(item, inner_t) for item in raw]
                return tuple(converted) if isinstance(raw, tuple) else converted
            return raw

        # Dict[str, T]
        if origin is dict and args and issubclass(args[0], str):
            if isinstance(raw, dict):
                inner_t = args[1]
                return {k: self._deep_convert(v, inner_t) for k, v in raw.items()}
            return raw

        # Dataclass itself
        if is_dataclass(target_type) and issubclass(target_type, Dataclass):
            if isinstance(raw, dict):
                # NEVER forward internal or alien keys
                allowed = {f.name: raw[f.name]
                           for f in fields(target_type)
                           if f.name in raw}
                return target_type(**allowed)
            return raw

        # Dict-like custom type (e.g. CfgNode)
        if (
            isinstance(target_type, type)
            and not is_dataclass(target_type)
            and issubclass(target_type, dict)
            and isinstance(raw, dict)
            and not isinstance(raw, target_type)
        ):
            try:
                return target_type(raw)
            except Exception:
                return raw

        return raw

    # ------------------------------------------------------------------
    # Recursive in-place update
    # ------------------------------------------------------------------
    _FieldPairs = typing.Union[
        Dict[str, Any],
        "Dataclass",
        Iterable[Tuple[str, Any]],
    ]

    def update(self, source: _FieldPairs = (), /, **kw: Any) -> None:
        """
        Like dict.update() with recursive semantics for nested Dataclass instances.
        """
        # Normalize
        if isinstance(source, Dataclass):
            updates = dict(source)
        elif isinstance(source, dict):
            updates = source
        else:
            updates = dict(source)
        updates.update(kw)

        # Apply
        for key, value in updates.items():
            current = getattr(self, key, None)

            # Nested Dataclass → recurse
            if isinstance(current, Dataclass) and isinstance(value, (dict, Dataclass)):
                patch = dict(value) if isinstance(value, Dataclass) else value
                current.update(**patch)

            # Ordinary attribute / plain overwrite
            else:
                setattr(self, key, value)
                self[key] = value

        # Refresh guarantees any fresh dict turns into Dataclass
        self._hydrate_recursive()

    def dump(self):
        def _to_clean_dict(obj):
            if isinstance(obj, (list, tuple)):
                return [_to_clean_dict(v) for v in obj]
            if hasattr(obj, "items"):
                return {
                    k: _to_clean_dict(v)
                    for k, v in obj.items()
                    if k not in ("__defaults__", "_already_hydrated")  # 过滤内部字段
                }
            return obj
        return json.dumps(_to_clean_dict(self), indent=4, ensure_ascii=False)

    def save(self, path: str) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.dump(), encoding="utf-8")
