from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

DEFAULT_HEAD = "EVENT"


def format_head_keys(keys: Iterable[Any]) -> str:
    return "[" + ", ".join(sorted(repr(key) for key in keys)) + "]"


def normalize_class_heads(class_labels: Any) -> tuple[dict[str, list[str]], dict[str, float], bool]:
    """Return ``head -> labels``, ``head -> loss weight``, and whether user passed a dict."""

    if isinstance(class_labels, Mapping):
        heads: dict[str, list[str]] = {}
        weights: dict[str, float] = {}
        for head, spec in class_labels.items():
            if not isinstance(head, str) or not head or "." in head or "/" in head:
                raise ValueError("classification head names must be non-empty strings without '.' or '/'")
            if not isinstance(spec, Mapping):
                raise ValueError(f"class_labels[{head!r}] must be a dict with 'name' and 'lambda'")
            if "name" not in spec or "lambda" not in spec:
                raise ValueError(f"class_labels[{head!r}] must define both 'name' and 'lambda'")
            names = spec["name"]
            if isinstance(names, (str, bytes)) or not isinstance(names, Sequence) or not names:
                raise ValueError(f"class_labels[{head!r}]['name'] must be a non-empty sequence")
            heads[head] = [str(name) for name in names]
            try:
                weight = float(spec["lambda"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"class_labels[{head!r}]['lambda'] must be a finite non-negative number"
                ) from exc
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"class_labels[{head!r}]['lambda'] must be a finite non-negative number")
            weights[head] = weight
        if not heads:
            raise ValueError("class_labels must define at least one classification head")
        return heads, weights, True

    if isinstance(class_labels, (str, bytes)) or not isinstance(class_labels, Sequence) or not class_labels:
        raise ValueError("class_labels must be a non-empty list or a head dictionary")
    return {DEFAULT_HEAD: [str(name) for name in class_labels]}, {DEFAULT_HEAD: 1.0}, False


def normalize_head_tensors(value: Any, head_names: Sequence[str], what: str) -> dict[str, torch.Tensor]:
    if isinstance(value, Mapping):
        keys = set(value.keys())
        expected = set(head_names)
        if keys != expected:
            raise ValueError(
                f"{what} heads {format_head_keys(keys)} do not match class_labels heads {format_head_keys(expected)}"
            )
        result = {head: torch.as_tensor(value[head]).long() for head in head_names}
    else:
        if len(head_names) != 1:
            raise ValueError(f"{what} must be a head dictionary when class_labels defines multiple heads")
        result = {head_names[0]: torch.as_tensor(value).long()}

    for head, tensor in result.items():
        if tensor.dim() != 1:
            raise ValueError(f"{what}[{head!r}] must be a 1D class-index tensor")
    return result


def normalize_head_config(
    value: Any,
    head_names: Sequence[str],
    what: str,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if value is None:
        if allow_empty:
            return {head: {} for head in head_names}
        raise ValueError(f"{what} must be explicitly provided")
    if allow_empty and value == {}:
        return {head: {} for head in head_names}

    if isinstance(value, Mapping) and set(value.keys()) == set(head_names):
        return {head: value[head] for head in head_names}

    if len(head_names) == 1:
        return {head_names[0]: value}

    raise ValueError(f"{what} must be a dictionary with exactly these heads: {list(head_names)}")
