"""Translate historical PI05 method names at configuration-loading boundaries.

The model and saved configurations use OPG/CCR names exclusively. This adapter
accepts old checkpoint JSON, training configuration files and command-line flags
without adding legacy fields to the public PI05 dataclass.
"""

import json
import sys
import tempfile
from pathlib import Path

import draccus
from draccus import cfgparsing

PI05_FIELD_ALIASES = {
    "causal_scale": "opg_scale",
    "use_cdsr": "use_ccr",
    "cdsr_layers": "ccr_layers",
    "cdsr_beta": "ccr_beta",
    "cdsr_dv": "ccr_top_k",
    "cdsr_eps": "ccr_eps",
    "cdsr_eval_only": "ccr_eval_only",
    "cdsr_reference_model_path": "ccr_reference_model_path",
}


def _option_value(args, name):
    for i, arg in enumerate(args):
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
        if arg == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _rename_fields(config):
    migrated = dict(config)
    for old, new in PI05_FIELD_ALIASES.items():
        if old in migrated:
            value = migrated.pop(old)
            if new in migrated and migrated[new] != value:
                raise ValueError(f"Conflicting PI05 configuration fields: {old} and {new}; use {new}")
            migrated[new] = value
    return migrated


def _rename_args(args, prefix):
    migrated = list(args)
    for old, new in PI05_FIELD_ALIASES.items():
        old_flag, new_flag = f"--{prefix}{old}", f"--{prefix}{new}"
        old_value = _option_value(args, old_flag)
        new_value = _option_value(args, new_flag)
        if old_value is not None and new_value is not None and old_value != new_value:
            raise ValueError(f"Conflicting PI05 arguments: {old_flag} and {new_flag}; use {new_flag}")
        migrated = [
            new_flag + arg[len(old_flag) :] if arg == old_flag or arg.startswith(old_flag + "=") else arg
            for arg in migrated
        ]
    return migrated


def parse_config(config_class, config_path=None, args=None):
    """Parse with PI05 aliases normalized before CLI overrides are merged.

    Both ``--key=value`` and ``--key value`` forms are supported. Other policy
    types retain their original fields. Source configuration files are read only.
    """
    args = list(sys.argv[1:] if args is None else args)
    # Resolve the same config_path override as draccus before inspecting the file.
    effective_path = _option_value(args, "--config_path") or config_path
    filtered = []
    skip_value = False
    for arg in args:
        if skip_value:
            skip_value = False
        elif arg == "--config_path":
            skip_value = True
        elif not arg.startswith("--config_path="):
            filtered.append(arg)
    args = filtered
    raw = {}
    if effective_path is not None:
        with open(effective_path) as stream:
            raw = cfgparsing.load_config(stream, file=effective_path)

    nested = "policy" in getattr(config_class, "__dataclass_fields__", {})
    prefix = "policy." if nested else ""
    policy = (raw.get("policy") or {}) if nested else raw
    default_type = "pi05" if config_class.__name__ == "PI05Config" else None
    policy_type = _option_value(args, f"--{prefix}type") or policy.get("type", default_type)
    if policy_type != "pi05":
        return draccus.parse(config_class, config_path=effective_path, args=args)

    args = _rename_args(args, prefix)
    migrated_policy = _rename_fields(policy)
    if migrated_policy == policy:
        return draccus.parse(config_class, config_path=effective_path, args=args)
    migrated = dict(raw, policy=migrated_policy) if nested else migrated_policy
    # A temporary JSON avoids rewriting user checkpoints or leaving migrated files behind.
    with tempfile.TemporaryDirectory(prefix="lerobot-pi05-config-") as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(migrated), encoding="utf-8")
        return draccus.parse(config_class, config_path=path, args=args)
