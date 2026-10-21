"""Real config/CLI round trips for the OPG and CCR naming migration."""

import json
import sys
from dataclasses import dataclass

import draccus
import pytest

from lerobot.configs import parser
from lerobot.configs.pi05_compat import PI05_FIELD_ALIASES, parse_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.ccr import apply_ccr
from lerobot.policies.pi05.cofact import orthogonal_guidance, reduce_counterfactual_covariance
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.opg import apply_opg


def as_legacy(config):
    raw = draccus.encode(config)
    for old, new in PI05_FIELD_ALIASES.items():
        raw[old] = raw.pop(new)
    return raw


def test_legacy_checkpoint_preserves_values_and_saves_only_paper_names(tmp_path):
    original = PI05Config(
        device="cpu",
        opg_scale=3,
        use_ccr=True,
        ccr_layers=[2, 5],
        ccr_beta=0.3,
        ccr_top_k=2,
        ccr_eps=2e-5,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(as_legacy(original)))
    before = path.read_bytes()
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert draccus.encode(loaded) == draccus.encode(original)
    assert path.read_bytes() == before
    saved = tmp_path / "new"
    loaded.save_pretrained(saved)
    raw = json.loads((saved / "config.json").read_text())
    assert not set(raw).intersection(PI05_FIELD_ALIASES)
    assert set(PI05_FIELD_ALIASES.values()).issubset(raw)
    assert draccus.encode(PI05Config.from_pretrained(saved)) == raw


@pytest.mark.parametrize("legacy_file", [False, True])
@pytest.mark.parametrize("legacy_cli", [False, True])
def test_checkpoint_cli_overrides_work_with_both_naming_generations(tmp_path, legacy_file, legacy_cli):
    config = PI05Config(device="cpu", opg_scale=2, ccr_beta=0.15)
    raw = as_legacy(config) if legacy_file else draccus.encode(config)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    args = (
        ["--causal_scale=1", "--use_cdsr=false", "--cdsr_beta=0.2"]
        if legacy_cli
        else ["--opg_scale=1", "--use_ccr=false", "--ccr_beta=0.2"]
    )
    loaded = PreTrainedConfig.from_pretrained(tmp_path, cli_overrides=args)
    assert loaded.opg_scale == 1 and not loaded.use_ccr and loaded.ccr_beta == 0.2


@pytest.mark.parametrize("legacy_file", [False, True])
def test_resume_training_config_migrates_nested_policy(tmp_path, legacy_file):
    raw = {
        "dataset": {"repo_id": "example/data"},
        "policy": as_legacy(PI05Config(device="cpu"))
        if legacy_file
        else draccus.encode(PI05Config(device="cpu")),
        "resume": True,
    }
    path = tmp_path / "train_config.json"
    path.write_text(json.dumps(raw))
    loaded = TrainPipelineConfig.from_pretrained(path, cli_args=["--policy.causal_scale=1"])
    assert loaded.resume and loaded.policy.opg_scale == 1
    encoded = loaded.to_dict()["policy"]
    assert not set(encoded).intersection(PI05_FIELD_ALIASES)


@pytest.mark.parametrize("separate_values", [False, True])
def test_training_cli_accepts_legacy_names(monkeypatch, separate_values):
    args = [
        "--dataset.repo_id=example/data",
        "--policy.type=pi05",
        "--policy.device=cpu",
        "--policy.causal_scale=1",
        "--policy.use_cdsr=false",
        "--policy.cdsr_layers=[2,3]",
    ]
    if separate_values:
        args = [part for arg in args for part in arg.split("=", 1)]
    monkeypatch.setattr(sys, "argv", ["train", *args])

    @parser.wrap()
    def read_config(cfg: TrainPipelineConfig):
        return cfg

    cfg = read_config()
    assert cfg.policy.opg_scale == 1 and not cfg.policy.use_ccr and cfg.policy.ccr_layers == [2, 3]


def test_legacy_config_path_argument_and_canonical_override(monkeypatch, tmp_path):
    path = tmp_path / "training.json"
    path.write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "example/data"},
                "policy": as_legacy(PI05Config(device="cpu")),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["train", f"--config_path={path}", "--policy.opg_scale=1"])

    @parser.wrap()
    def read_config(cfg: TrainPipelineConfig):
        return cfg

    assert read_config().policy.opg_scale == 1


def test_conflicting_names_fail_explicitly(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "device": "cpu",
                "causal_scale": 1,
                "opg_scale": 2,
            }
        )
    )
    with pytest.raises(ValueError, match="Conflicting PI05 configuration"):
        PreTrainedConfig.from_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Conflicting PI05 arguments"):
        parse_config(PI05Config, args=["--opg_scale=2", "--causal_scale=1"])


def test_unrelated_configuration_is_not_renamed():
    @dataclass
    class UnrelatedConfig:
        causal_scale: float = 5

    assert parse_config(UnrelatedConfig, args=["--causal_scale=7"]).causal_scale == 7


def test_pi0_historical_experiment_keeps_its_own_names():
    cfg = parse_config(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=example/data",
            "--policy.type=pi0",
            "--policy.device=cpu",
            "--policy.causal_scale=1",
            "--policy.use_cdsr=false",
            "--policy.cdsr_beta=0.2",
        ],
    )
    assert cfg.policy.type == "pi0" and cfg.policy.cdsr_beta == 0.2
    assert "cdsr_beta" in draccus.encode(cfg.policy)
    assert "ccr_beta" not in draccus.encode(cfg.policy)


def test_old_helper_imports_refer_to_same_operations():
    assert orthogonal_guidance is apply_opg
    assert reduce_counterfactual_covariance is apply_ccr
