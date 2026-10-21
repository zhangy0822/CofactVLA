#!/usr/bin/env python3
"""Evaluate a local PI05 checkpoint on PiPer; hardware is enabled only with --execute."""

import argparse
import csv
import json
import select
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--camera-config", required=True, help="JSON mapping cam_high/cam_wrist to device serials"
    )
    parser.add_argument("--can_port", default="can0")
    parser.add_argument("--method", choices=["cofact", "baseline"], default="cofact")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=1000,
        help="Operational step limit; not specified in the paper",
    )
    parser.add_argument("--results", default="outputs/piper/trials.csv")
    parser.add_argument("--execute", action="store_true", help="Load policy and enable the real robot")
    args = parser.parse_args()
    if args.num_episodes < 1 or args.steps_per_episode < 1:
        parser.error("Episode count and step limit must be positive")
    return args


def main():
    args = parse_args()
    if not args.execute:
        print(
            json.dumps(
                {**vars(args), "fps": 10, "action_chunk_size": 50, "executed_steps_per_chunk": 10}, indent=2
            )
        )
        print("Preview only. Add --execute on the Linux robot host after configuring cameras and CAN.")
        return
    if sys.platform != "linux":
        raise RuntimeError("PiPer execution requires Linux, SocketCAN and an NVIDIA CUDA GPU")
    checkpoint = Path(args.policy_path).expanduser().resolve()
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    camera_serials = json.loads(Path(args.camera_config).read_text())
    if set(camera_serials) != {"cam_high", "cam_wrist"} or not all(camera_serials.values()):
        raise ValueError("Provide both cam_high and cam_wrist camera serials")
    if len(set(camera_serials.values())) != 2 or any("REPLACE" in s for s in camera_serials.values()):
        raise ValueError("Replace camera placeholders with two distinct device serials")
    if not sys.stdin.isatty():
        raise RuntimeError("Supervised trials require an interactive terminal")
    results = Path(args.results)
    if results.exists():
        raise FileExistsError(f"Refusing to overwrite trial results: {results}")

    # Lazy imports allow --help and preview on a Mac without installing robot dependencies.
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.processor import make_default_processors
    from lerobot.robots.piper import Piper, PiperConfig
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.control_utils import predict_action

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this deployment configuration")
    robot_cfg = PiperConfig(can_port=args.can_port, camera_serials=camera_serials, fps=10)
    robot = Piper(robot_cfg)
    action_processor, _, obs_processor = make_default_processors()
    ds_features = {
        **aggregate_pipeline_dataset_features(
            pipeline=obs_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
        **aggregate_pipeline_dataset_features(
            pipeline=action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
    }
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type != "pi05":
        raise ValueError("This deployment script supports PI05 checkpoints only")
    cfg.device = "cuda"
    cfg.pretrained_path = str(checkpoint)
    cfg.compile_model = False
    cfg.n_action_steps = 10
    cfg.opg_scale = 2.0 if args.method == "cofact" else 1.0
    cfg.use_ccr = args.method == "cofact"
    cfg.ccr_layers = [15, 16]
    cfg.ccr_beta = 0.15
    cfg.ccr_top_k = 4
    cfg.ccr_eps = 1e-5
    cfg.ccr_eval_only = True
    expected_images = {key for key in cfg.input_features if key.startswith("observation.images.")}
    actual_images = {key for key in ds_features if key.startswith("observation.images.")}
    if expected_images != actual_images:
        raise ValueError(f"Checkpoint camera keys {expected_images} differ from robot keys {actual_images}")
    if cfg.output_features[ACTION].shape != (7,):
        raise ValueError("PiPer requires 6 joint positions in radians and one normalized gripper action")
    policy = get_policy_class("pi05").from_pretrained(checkpoint, config=cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    results.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    completed = 0
    try:
        # connect() also cleans up partially connected resources on failure.
        robot.connect()
        with results.open("x", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["episode", "task", "method", "success", "steps", "seconds"]
            )
            writer.writeheader()
            for episode in range(1, args.num_episodes + 1):
                input(
                    f"Reset task scene, clear the workspace, then press Enter for trial {episode}/{args.num_episodes}: "
                )
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                steps = 0
                started = time.perf_counter()
                # Reobserve after each 10-action chunk. Inference adds a pause;
                # the 10 Hz target applies to action dispatch, not wall-clock throughput.
                print("Press Enter to end this trial and score it; Ctrl+C aborts the session.")
                end_trial = False
                while steps < args.steps_per_episode and not end_trial:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        break
                    obs = build_dataset_frame(ds_features, robot.get_observation(), prefix=OBS_STR)
                    chunk = []
                    for _ in range(min(10, args.steps_per_episode - steps)):
                        action = predict_action(
                            observation=obs,
                            policy=policy,
                            device=torch.device("cuda"),
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            use_amp=cfg.use_amp,
                            task=args.task,
                            robot_type=robot.robot_type,
                        )
                        chunk.append(make_robot_action(action, ds_features))
                    for action in chunk:
                        if select.select([sys.stdin], [], [], 0)[0]:
                            sys.stdin.readline()
                            end_trial = True
                            break
                        tick = time.perf_counter()
                        robot.send_action(action)
                        steps += 1
                        time.sleep(max(0, 0.1 - (time.perf_counter() - tick)))
                verdict = ""
                while verdict not in {"y", "n"}:
                    verdict = input("Task succeeded? [y/n]: ").strip().lower()
                success = int(verdict == "y")
                writer.writerow(
                    dict(
                        episode=episode,
                        task=args.task,
                        method=args.method,
                        success=success,
                        steps=steps,
                        seconds=round(time.perf_counter() - started, 3),
                    )
                )
                stream.flush()
                successes += success
                completed += 1
                print(f"Success: {successes}/{completed} ({100 * successes / completed:.1f}%)")
    except KeyboardInterrupt:
        print("Interrupted. Completed trials remain in the CSV; this is a partial evaluation.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
