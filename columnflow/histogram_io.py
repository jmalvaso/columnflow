# coding: utf-8

"""Storage helpers for legacy per-variable and bundled histogram files."""

from __future__ import annotations

import re


def uses_histogram_bundles(config_inst):
    return callable(config_inst.x("histogram_storage_groups", None))


def histogram_storage_groups(config_inst, variables, dataset=None):
    """Resolve requested variables to complete, deterministic storage groups."""
    variables = tuple(variables)
    resolver = config_inst.x("histogram_storage_groups", None)
    if not callable(resolver):
        return {variable: (variable,) for variable in variables}

    groups = resolver(variables, dataset=dataset)
    resolved = {}
    seen = set()
    for key, group_variables in sorted(groups.items()):
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", key):
            raise ValueError(f"invalid histogram storage group key: {key!r}")
        group_variables = tuple(sorted(set(group_variables)))
        if not group_variables or seen.intersection(group_variables):
            raise ValueError(f"empty or overlapping histogram storage group: {key!r}")
        resolved[key] = group_variables
        seen.update(group_variables)
    if not set(variables).issubset(seen):
        raise ValueError(f"histogram storage groups omit requested variables: {set(variables) - seen}")
    return resolved


def histogram_target(collection, variable, config_inst, dataset=None):
    """Return a legacy variable target or the unique bundle containing it."""
    if variable in collection.targets:
        return collection.targets[variable], False

    groups = histogram_storage_groups(config_inst, (variable,), dataset=dataset)
    key = next(key for key, variables in groups.items() if variable in variables)
    return collection.targets[key], True


def load_histogram(collection, variable, config_inst, dataset=None):
    """Load one variable without exposing the physical storage layout to readers."""
    target, bundled = histogram_target(collection, variable, config_inst, dataset=dataset)
    payload = target.load(formatter="pickle")
    return payload[variable] if bundled else payload