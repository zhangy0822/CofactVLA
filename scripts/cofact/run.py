#!/usr/bin/env python3
"""Portable paper-parameter launchers; --dry-run requires only the Python stdlib."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def build_command(args):
    train = args.command in {"train-libero", "train-real"}
    method = [
        f"--policy.opg_scale={2.0 if args.method == 'cofact' else 1.0}",
        f"--policy.use_ccr={str(args.method == 'cofact').lower()}",
        "--policy.ccr_layers=[15,16]",
        "--policy.ccr_beta=0.15",
        "--policy.ccr_top_k=4",
        "--policy.ccr_eps=1e-5",
        "--policy.ccr_eval_only=true",
    ]
    if train:
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for training")
        if args.command == "train-real" and not args.dataset_repo_id:
            raise ValueError("--dataset-repo-id is required for real-world training")
        repo_id = args.dataset_repo_id or "lerobot/libero"
        pretrained = args.pretrained or (
            "lerobot/pi05_libero_base" if args.command == "train-libero" else "lerobot/pi05_base"
        )
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes=4",
            "--num_machines=1",
            "--machine_rank=0",
            "--mixed_precision=no",
            "--dynamo_backend=no",
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={Path(args.dataset_root).expanduser().resolve()}",
            "--dataset.video_backend=pyav",
            "--policy.type=pi05",
            f"--policy.pretrained_path={pretrained}",
            "--policy.device=cuda",
            "--policy.dtype=bfloat16",
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
            "--policy.freeze_vision_encoder=false",
            "--policy.train_expert_only=false",
            "--policy.gradient_checkpointing=true",
            "--policy.compile_model=true",
            "--policy.chunk_size=50",
            "--policy.optimizer_lr=2.5e-5",
            "--policy.optimizer_weight_decay=0.01",
            "--policy.optimizer_betas=[0.9,0.95]",
            "--policy.scheduler_warmup_steps=1000",
            "--policy.scheduler_decay_steps=6000",
            "--steps=6000",
            "--batch_size=32",
            "--save_freq=1000",
            "--eval_freq=0",
        ]
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluation")
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            "--env.type=libero",
            "--env.task=libero_spatial,libero_object,libero_goal,libero_10",
            "--env.max_parallel_tasks=1",
            "--eval.batch_size=1",
            "--eval.n_episodes=10",
            f"--policy.path={Path(args.checkpoint).expanduser().resolve()}",
            "--policy.device=cuda",
            "--policy.dtype=bfloat16",
            "--policy.compile_model=false",
            "--policy.n_action_steps=10",
        ]
    return cmd + method + [f"--output_dir={Path(args.output).expanduser().resolve()}", f"--seed={args.seed}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train-libero", "eval-libero", "train-real"])
    parser.add_argument("--method", choices=["cofact", "baseline"], default="cofact")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-repo-id")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Initial checkpoint directory or Hub ID; use the exact experiment asset if available",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--seed", type=int, default=1000, help="Inherited LeRobot default; paper does not specify seed"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        cmd = build_command(args)
    except ValueError as e:
        parser.error(str(e))
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3" if args.command.startswith("train-") else "0")
    if args.command == "eval-libero":
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
    print("CUDA_VISIBLE_DEVICES=" + shlex.quote(env["CUDA_VISIBLE_DEVICES"]), flush=True)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if sys.platform != "linux":
        parser.error("Execution requires a Linux NVIDIA GPU host. Use --dry-run for local inspection.")
    if Path(args.output).exists():
        parser.error("Output directory already exists; choose a new directory to preserve previous runs")
    if args.command.startswith("train-") and not (Path(args.dataset_root) / "meta/info.json").is_file():
        parser.error("Dataset root must contain a LeRobot dataset with meta/info.json")
    if args.command.startswith("train-"):
        meta_root = Path(args.dataset_root) / "meta"
        info = json.loads((meta_root / "info.json").read_text())
        if info.get("codebase_version") != "v3.0":
            parser.error("Convert this dataset to LeRobot v3.0 first; see docs/release/INSTALL.md")
        stats = json.loads((meta_root / "stats.json").read_text())
        for key in ("observation.state", "action"):
            if not {"q01", "q99"} <= stats.get(key, {}).keys():
                parser.error(f"Missing quantile statistics for {key}; see docs/release/INSTALL.md")
    import torch

    needed = 4 if args.command.startswith("train-") else 1
    if not torch.cuda.is_available() or torch.cuda.device_count() != needed:
        parser.error(f"Expected {needed} visible CUDA GPU(s); check CUDA_VISIBLE_DEVICES")
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
