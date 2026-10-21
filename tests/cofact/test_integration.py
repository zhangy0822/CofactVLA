"""Small architecture and launcher checks without pretrained weights or hardware."""

from argparse import Namespace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest
import torch
from transformers import GemmaConfig

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, make_att_2d_masks
from lerobot.policies.pi_gemma import PiGemmaModel
from lerobot.robots.piper import Piper, PiperConfig

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cofact_run", ROOT / "scripts/cofact/run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_training_command_parses_and_keeps_paper_warmup():
    args = Namespace(
        command="train-libero",
        method="cofact",
        dataset_root="data/libero",
        dataset_repo_id=None,
        pretrained=None,
        output="outputs/unused",
        seed=1000,
    )
    cmd = launcher.build_command(args)
    cfg = draccus.parse(TrainPipelineConfig, args=cmd[cmd.index("lerobot.scripts.lerobot_train") + 1 :])
    assert cfg.steps == 6000 and cfg.batch_size == 32
    assert cfg.policy.ccr_layers == [15, 16]
    optimizer = cfg.policy.get_optimizer_preset().build([torch.nn.Parameter(torch.ones(1))])
    scheduler = cfg.policy.get_scheduler_preset().build(optimizer, cfg.steps)
    # At step 200 we must still be warming up (the old implicit rescaling ended at 200).
    assert scheduler.lr_lambdas[0](200) < 0.21
    assert cfg.policy.scheduler_warmup_steps == 1000
    assert cfg.policy.scheduler_decay_steps == 6000
    assert str(cfg.policy.pretrained_path) == "lerobot/pi05_libero_base"
    assert not cfg.policy.push_to_hub and not cfg.wandb.enable


def tiny_prefix():
    cfg = GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    prefix = PiGemmaModel(cfg).eval()
    wrapper = PI05Pytorch.__new__(PI05Pytorch)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = PI05Config(device="cpu", ccr_layers=[1])
    wrapper.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=prefix))
    )
    return wrapper, prefix


def test_ccr_zero_strength_matches_standard_prefix_cache():
    torch.manual_seed(1)
    wrapper, prefix = tiny_prefix()
    embs = torch.randn(2, 5, 16)
    pads = torch.tensor([[True] * 5, [True, True, True, False, False]])
    positions = pads.long().cumsum(1) - 1
    attention = wrapper._prepare_attention_masks_4d(make_att_2d_masks(pads, torch.zeros_like(pads)))
    with torch.no_grad():
        standard = prefix(
            inputs_embeds=embs, attention_mask=attention, position_ids=positions, use_cache=True
        ).past_key_values
        manual = wrapper._build_ccr_prefix_cache(embs, pads, attention, positions, 1, beta=0)
    assert len(manual.layers) == 3
    for a, b in zip(standard.layers, manual.layers, strict=True):
        torch.testing.assert_close(a.keys, b.keys)
        torch.testing.assert_close(a.values, b.values)


def test_ccr_only_intercepts_selected_layers_and_rejects_invalid_index():
    wrapper, _ = tiny_prefix()
    calls = []

    def record(features, *args):
        calls.append(features.shape)
        return features

    wrapper._ccr_projection = record
    embs = torch.randn(2, 4, 16)
    pads = torch.ones(2, 4).bool()
    positions = pads.long().cumsum(1) - 1
    attention = wrapper._prepare_attention_masks_4d(make_att_2d_masks(pads, torch.zeros_like(pads)))
    wrapper._build_ccr_prefix_cache(embs, pads, attention, positions, 1)
    assert len(calls) == 2  # keys and values of the one selected layer
    wrapper.config.ccr_layers = [3]
    with pytest.raises(ValueError):
        wrapper._build_ccr_prefix_cache(embs, pads, attention, positions, 1)


def test_piper_clamps_finite_actions_and_rejects_nan(tmp_path):
    robot = Piper(
        PiperConfig(camera_serials={"cam_high": "one", "cam_wrist": "two"}, calibration_dir=tmp_path)
    )

    class SDK:
        def MotionCtrl_2(self, *args):
            pass

        def JointCtrl(self, *args):
            self.joints = args

        def GripperCtrl(self, *args):
            self.gripper = args

    robot._sdk = SDK()
    robot._connected = True
    action = {f"joint_{i}.pos": 100.0 for i in range(1, 7)} | {"gripper.pos": 2.0}
    sent = robot.send_action(action)
    assert sent["joint_3.pos"] == 0 and sent["gripper.pos"] == 1
    assert robot._sdk.gripper[0] == 70000
    with pytest.raises(ValueError):
        robot.send_action(action | {"joint_1.pos": float("nan")})


def test_piper_can_clean_partial_connection(tmp_path):
    robot = Piper(PiperConfig(calibration_dir=tmp_path))
    stopped = []
    robot._pipelines = {"test": SimpleNamespace(stop=lambda: stopped.append(True))}
    robot.disconnect()
    robot.disconnect()
    assert stopped == [True] and not robot.is_connected


def test_failed_checkpoint_loading_raises_instead_of_returning_random_model(monkeypatch):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import transformers.utils

    def fake_init(self, config, **kwargs):
        torch.nn.Module.__init__(self)

    def missing_file(*args, **kwargs):
        raise OSError("test missing checkpoint")

    monkeypatch.setattr(PI05Policy, "__init__", fake_init)
    monkeypatch.setattr(transformers.utils, "cached_file", missing_file)
    with pytest.raises(RuntimeError, match="Failed to load PI05 weights"):
        PI05Policy.from_pretrained("missing-local-checkpoint", config=PI05Config(device="cpu"))


@pytest.mark.parametrize("method", ["baseline", "cofact"])
def test_tiny_pi05_training_and_sampling_without_weights(monkeypatch, method):
    """Exercise complete forward/backward and both inference cache paths on CPU."""
    from transformers import PaliGemmaConfig, SiglipVisionConfig
    from lerobot.policies.pi05 import modeling_pi05 as modeling

    def small_paligemma():
        return PaliGemmaConfig(
            vision_config=SiglipVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=16,
                patch_size=8,
            ),
            text_config=GemmaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            ),
        )

    original_paligemma = modeling.PaliGemmaForConditionalGenerationWithPiGemma

    def small_projection_model(config):
        # Production fixes the visual projection at width 2048; match the tiny fixture's width.
        config.vision_config.projection_dim = config.text_config.hidden_size
        return original_paligemma(config=config)

    monkeypatch.setattr(modeling, "PaliGemmaForConditionalGenerationWithPiGemma", small_projection_model)
    monkeypatch.setattr(modeling, "CONFIG_MAPPING", {"paligemma": small_paligemma, "gemma": GemmaConfig})
    monkeypatch.setattr(modeling, "get_gemma_config", lambda _: modeling.GemmaConfig(16, 2, 32, 8, 1, 2))
    cfg = PI05Config(
        device="cpu",
        dtype="float32",
        image_resolution=(16, 16),
        max_action_dim=7,
        chunk_size=3,
        n_action_steps=3,
        num_inference_steps=2,
        opg_scale=2 if method == "cofact" else 1,
        use_ccr=method == "cofact",
        ccr_layers=[1],
    )
    model = PI05Pytorch(cfg)
    images = [torch.randn(1, 3, 16, 16)]
    image_masks = [torch.ones(1, dtype=torch.bool)]
    tokens = torch.tensor([[1, 2, 3]])
    masks = torch.ones_like(tokens, dtype=torch.bool)
    actions = torch.randn(1, 3, 7)
    loss = model(images, image_masks, tokens, masks, actions)
    assert loss.shape == actions.shape and torch.isfinite(loss).all()
    loss.mean().backward()
    assert torch.isfinite(model.action_out_proj.weight.grad).all()
    model.eval()
    sampled = model.sample_actions(images, image_masks, tokens, masks)
    assert sampled.shape == actions.shape and torch.isfinite(sampled).all()


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_eval_launcher_overrides_saved_configuration(monkeypatch, tmp_path, legacy_checkpoint):
    import json
    import sys
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.pi05_compat import PI05_FIELD_ALIASES

    checkpoint = tmp_path / "checkpoint"
    PI05Config(device="cpu", opg_scale=1, use_ccr=False).save_pretrained(checkpoint)
    if legacy_checkpoint:
        path = checkpoint / "config.json"
        raw = json.loads(path.read_text())
        for old, new in PI05_FIELD_ALIASES.items():
            raw[old] = raw.pop(new)
        path.write_text(json.dumps(raw))
    args = Namespace(
        command="eval-libero",
        method="cofact",
        checkpoint=str(checkpoint),
        output=str(tmp_path / "evaluation"),
        seed=1000,
    )
    cmd = launcher.build_command(args)
    monkeypatch.setattr(sys, "argv", ["eval", *cmd[cmd.index("lerobot.scripts.lerobot_eval") + 1 :]])

    @parser.wrap()
    def read_config(cfg: EvalPipelineConfig):
        return cfg

    cfg = read_config()
    assert cfg.eval.n_episodes == 10 and cfg.eval.batch_size == 1
    assert cfg.policy.use_ccr and cfg.policy.opg_scale == 2
    assert cfg.policy.ccr_layers == [15, 16] and cfg.policy.n_action_steps == 10
    assert cfg.env.task == "libero_spatial,libero_object,libero_goal,libero_10"
