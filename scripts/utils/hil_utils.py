import os
import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaaclab.envs import DirectRLEnv

from scripts.eval_policy.base_policy import BasePolicy
from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)
from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .common import stabilize_garment_after_reset
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def run_hil_loop(
    env: DirectRLEnv,
    policy: BasePolicy,
    teleop_interface: Any,
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Human-in-the-Loop (HIL) evaluation loop.
    Implements the 3-step state machine:
    1. Autonomous (default)
    2. Paused (Space)
    3. Takeover (c)
    4. Resume (p) -> back to Autonomous
    """

    # --- Dataset Recording Setup ---
    eval_dataset = None
    json_path = None
    episode_index = 0
    if args.save_datasets:
        features = None
        if args.dataset_root and Path(args.dataset_root).exists():
            source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
            features = dict(source_dataset.meta.features)
            fps = source_dataset.fps
        else:
            fps = 30  # Default FPS
            action_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            if is_bimanual:
                left_names = [f"left_{n}" for n in action_names]
                right_names = [f"right_{n}" for n in action_names]
                joint_names = left_names + right_names
            else:
                joint_names = action_names
            dim = len(joint_names)
            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
                "action": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
            }
            image_keys = ["top_rgb", "left_rgb", "right_rgb"] if is_bimanual else ["top_rgb", "wrist_rgb"]
            for key in image_keys:
                features[f"observation.images.{key}"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                }
        
        # Add HIL-specific feature: boolean flag for human interventions
        features["action_is_intervention"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_intervention"],
        }

        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="lehome_hil",
            fps=fps,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=features,
        )
        json_path = eval_dataset.root / "meta" / "garment_info.json"

    # --- HIL State Machine Events ---
    events = {
        "policy_paused": False,
        "correction_active": False,
        "resume_policy": False,
        "exit_early": False, # 'N' or Right arrow to exit
        "abort": False       # 'ESC'
    }

    def on_pause():
        if not events["policy_paused"] and not events["correction_active"]:
            logger.info("[HIL] PAUSED - Press 'c' to take control, or 'p' to resume.")
            events["policy_paused"] = True

    def on_take_control():
        if events["policy_paused"] and not events["correction_active"]:
            logger.info("[HIL] TAKING CONTROL - Recording corrections...")
            events["correction_active"] = True

    def on_resume():
        if events["policy_paused"] or events["correction_active"]:
            logger.info("[HIL] RESUMING POLICY - Resetting policy buffer...")
            events["resume_policy"] = True

    def on_success():
        logger.info("[HIL] Marking episode as complete.")
        events["exit_early"] = True

    def on_abort():
        logger.info("[HIL] Aborting HIL session.")
        events["abort"] = True

    teleop_interface.add_callback("SPACE", on_pause)
    teleop_interface.add_callback("c", on_take_control)
    teleop_interface.add_callback("p", on_resume)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("ESCAPE", on_abort)

    all_episode_metrics = []
    logger.info(f"Starting HIL evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    for i in range(args.num_episodes):
        if events["abort"]:
            break

        # 1. Reset Environment, Policy, & Teleop
        env.reset()
        policy.reset()
        teleop_interface.reset()
        stabilize_garment_after_reset(env, args)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()

        episode_frames = (
            {k: [] for k in observation_dict.keys() if "images" in k}
            if args.save_video else {}
        )

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)

        # Reset episode-level events
        events["policy_paused"] = False
        events["correction_active"] = False
        events["resume_policy"] = False
        events["exit_early"] = False

        # Pre-warm the teleop interface
        teleop_interface.advance()

        last_action_tensor = None

        logger.info(f"[HIL] Episode {i+1} started. Autonomous mode active.")

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            if events["abort"]:
                break
            
            if events["exit_early"]:
                break

            # Handle Resume
            if events["resume_policy"]:
                events["resume_policy"] = False
                events["policy_paused"] = False
                events["correction_active"] = False
                policy.reset() # Critical: clear policy's action chunk buffer
                teleop_interface.reset()
                logger.info("[HIL] Resumed Autonomous mode.")

            # --- Action Selection ---
            if events["correction_active"]:
                # HUMAN TAKEOVER
                action_np = teleop_interface.advance()
                if action_np is not None:
                    action_tensor = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)
                    last_action_tensor = action_tensor
                else:
                    action_tensor = last_action_tensor # Maintain pose if no new input
            elif events["policy_paused"]:
                # PAUSED (Holding still)
                action_tensor = last_action_tensor
                # advance teleop interface so it matches real-world arm movement without sending to sim yet
                teleop_interface.advance() 
            else:
                # AUTONOMOUS
                action_np = policy.select_action(observation_dict)
                action_tensor = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)
                last_action_tensor = action_tensor
                # keep teleop interface synced just in case
                teleop_interface.advance()

            # --- IK (if needed) ---
            if action_tensor is not None and args.use_ee_pose and ee_solver is not None:
                current_joints = (
                    torch.from_numpy(observation_dict["observation.state"])
                    .float()
                    .to(args.device)
                )
                action_env = convert_ee_pose_to_joints(
                    ee_pose_action=action_tensor.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)
            else:
                action_env = action_tensor

            # --- Step Environment ---
            if action_env is not None:
                env.step(action_env)

            # Check success
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50 

            # Reward & length
            reward_value = env._get_rewards()
            if isinstance(reward_value, torch.Tensor):
                reward = reward_value.item()
            else:
                reward = float(reward_value)

            episode_return += reward
            if not success_flag:
                episode_length += 1

            observation_dict = env._get_observations()

            # --- Dataset Recording ---
            # Even if paused, we record frames if we want continuous data.
            # But usually we only record if it's autonomous or active correction.
            if args.save_datasets and not events["policy_paused"]:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                
                # Tag whether this frame was human intervention
                is_intervention = np.array([events["correction_active"]], dtype=np.bool_)
                frame["action_is_intervention"] = is_intervention

                eval_dataset.add_frame(frame)

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False

        if args.save_datasets and not events["abort"]:
            # In HIL, we typically save all episodes we don't abort/discard
            eval_dataset.save_episode()
            append_episode_initial_pose(
                json_path,
                episode_index,
                object_initial_pose,
                garment_name=garment_name,
            )
            episode_index += 1

        if args.save_video and not events["abort"]:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
                garment_name=garment_name,
            )

        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}"
        )

    if args.save_datasets and eval_dataset is not None:
        eval_dataset.clear_episode_buffer()
        eval_dataset.finalize()

    return all_episode_metrics

def hil_eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """
    Main entry point for HIL evaluation logic.
    """
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.eval_policy import PolicyRegistry
    from scripts.utils.dataset_record import create_teleop_interface

    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = False
    if args.use_random_seed:
        env_cfg.use_random_seed = True
    else:
        env_cfg.use_random_seed = False
        env_cfg.seed = args.seed
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed

    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # 2. Initialize Policy
    logger.info(f"Initializing Policy Type: {args.policy_type}")
    if not PolicyRegistry.is_registered(args.policy_type):
        available_policies = PolicyRegistry.list_policies()
        raise ValueError(
            f"Policy type '{args.policy_type}' not found in registry. "
            f"Available policies: {', '.join(available_policies)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    policy_kwargs = {"device": device}
    if args.policy_type == "lerobot":
        if not args.policy_path:
            raise ValueError("--policy_path is required for lerobot policy type")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for lerobot policy type")
        policy_kwargs.update(
            {
                "policy_path": args.policy_path,
                "dataset_root": args.dataset_root,
                "task_description": args.task_description,
            }
        )
    else:
        if args.policy_path:
            policy_kwargs["model_path"] = args.policy_path

    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    # 3. Initialize IK Solver (If needed)
    ee_solver = None
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics
        urdf_path = args.ee_urdf_path 
        joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll",
        ]
        ee_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )
        logger.info(f"IK solver loaded.")

    # 4. Load Evaluation List
    eval_list = []
    if args.garment_type == "custom":
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", "Release_test_list.txt"
        )
    else:
        type_map = {
            "top_long": "Top_Long",
            "top_short": "Top_Short",
            "pant_long": "Pant_Long",
            "pant_short": "Pant_Short",
        }
        file_prefix = type_map.get(args.garment_type, "Top_Long")
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", file_prefix, f"{file_prefix}.txt"
        )

    logger.info(f"Loading evaluation list for category '{args.garment_type}' from: {eval_list_path}")
    if not os.path.exists(eval_list_path):
        raise FileNotFoundError(f"Evaluation list not found: {eval_list_path}")

    with open(eval_list_path, "r") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
        for name in names:
            eval_list.append((name, "Release"))

    if not eval_list:
        raise ValueError(f"No garments found to evaluate for category '{args.garment_type}'.")

    # Init Env
    first_name, first_stage = eval_list[0]
    env_cfg.garment_name = first_name
    env_cfg.garment_version = first_stage
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    # Init Teleop Interface
    logger.info(f"Initializing Teleop Device: {args.teleop_device}")
    teleop_interface = create_teleop_interface(env, args)

    all_garment_metrics = []
    try:
        for garment_idx, (garment_name, garment_stage) in enumerate(eval_list):
            logger.info(
                f"Evaluating: {garment_name} ({garment_stage}) ({garment_idx+1}/{len(eval_list)})"
            )

            if garment_idx > 0:
                if hasattr(env, "switch_garment"):
                    env.switch_garment(garment_name, garment_stage)
                    env.reset()
                    policy.reset()
                else:
                    env.close()
                    env_cfg.garment_name = garment_name
                    env_cfg.garment_version = garment_stage
                    env = gym.make(args.task, cfg=env_cfg).unwrapped
                    env.initialize_obs()
                    policy.reset()
                    # Re-bind teleop to new env
                    teleop_interface = create_teleop_interface(env, args)

            metrics = run_hil_loop(
                env=env,
                policy=policy,
                teleop_interface=teleop_interface,
                args=args,
                ee_solver=ee_solver,
                is_bimanual=is_bimanual,
                garment_name=garment_name,
            )

            all_garment_metrics.append({"garment_name": garment_name, "metrics": metrics})

    finally:
        env.close()

    # Print summary across all garments
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    if all_garment_metrics:
        all_episodes = []
        for garment_data in all_garment_metrics:
            for episode_metric in garment_data["metrics"]:
                episode_metric["garment_name"] = garment_data["garment_name"]
                all_episodes.append(episode_metric)

        calculate_and_print_metrics(all_episodes)

        logger.info("=" * 60)
        logger.info("Per-Garment Summary")
        logger.info("=" * 60)
        for garment_data in all_garment_metrics:
            garment_name = garment_data["garment_name"]
            metrics = garment_data["metrics"]
            success_count = sum(1 for m in metrics if m["success"])
            success_rate = success_count / len(metrics) if metrics else 0.0
            avg_return = np.mean([m["return"] for m in metrics]) if metrics else 0.0
            logger.info(
                f"  {garment_name}: Success Rate = {success_rate:.2%}, Avg Return = {avg_return:.2f}"
            )
    else:
        logger.info("No metrics collected (all evaluations failed/aborted)")

    logger.info("=" * 60)
    logger.info("HIL Evaluation completed successfully")
    logger.info("=" * 60)
