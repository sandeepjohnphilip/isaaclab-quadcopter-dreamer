"""
WATCH SINGLE DRONE
==================

Watch a single drone navigate the corridor with step-by-step output.
Shows exactly what the drone is doing and whether it reaches the goal.

Usage:
    # Fixed obstacle, with rendering
    ./isaaclab.sh -p source/standalone/watch_drone.py \
        --checkpoint checkpoints_dreamer/best.pt \
        --obstacle

    # Random obstacle each run
    ./isaaclab.sh -p source/standalone/watch_drone.py \
        --checkpoint checkpoints_dreamer/best.pt \
        --random_obstacle

    # Headless + slower stepping for debugging
    ./isaaclab.sh -p source/standalone/watch_drone.py \
        --checkpoint checkpoints_dreamer/best.pt \
        --obstacle --headless --slow
"""

import argparse
import os
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--corridor_length", type=float, default=10.0)

# Obstacle options
parser.add_argument("--obstacle", action="store_true",
                    help="Enable the default obstacle at x=6m, y=0, size=0.8m")
parser.add_argument("--random_obstacle", action="store_true",
                    help="Enable a random obstacle (position and size sampled per run)")

# Visualization options
parser.add_argument("--slow", action="store_true",
                    help="Add delay between steps for visualization")
parser.add_argument("--view_delay", type=float, default=7.0,
                    help="Seconds to wait after reset so you can adjust the camera")

args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab_tasks.direct.quadcopter.quadcopter_corridor_env import CorridorEnv, CorridorEnvCfg
from isaaclab_tasks.direct.quadcopter.dreamerv2_simple import DreamerAgent


def watch():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        simulation_app.close()
        return

    # ---------------------------------
    # Environment configuration
    # ---------------------------------
    cfg = CorridorEnvCfg()
    cfg.scene.num_envs = 1
    cfg.corridor_length = args.corridor_length
    cfg.episode_length_s = 15.0  # Longer for visualization

    # Enable obstacle if requested
    cfg.obstacle_enabled = args.obstacle or args.random_obstacle

    if cfg.obstacle_enabled:
        if args.random_obstacle:
            # Sample a random, but reasonable obstacle inside the corridor
            # Keep some margins from start and goal so it’s actually in the middle region.
            margin_x = 2.0
            cfg.obstacle_x = random.uniform(
                margin_x,
                cfg.corridor_length - margin_x
            )

            # Lateral limits: keep obstacle inside corridor walls with a bit of padding
            half_w = cfg.corridor_width / 2.0
            lateral_margin = 0.3
            cfg.obstacle_y = random.uniform(
                - (half_w - lateral_margin),
                + (half_w - lateral_margin)
            )

            # Reasonable size range
            cfg.obstacle_size = random.uniform(0.6, 1.2)
        else:
            # Default “normal” obstacle
            cfg.obstacle_x = 6.0
            cfg.obstacle_y = 0.0
            cfg.obstacle_size = 0.8

    # Success threshold used inside the env (unchanged)
    cfg.goal_threshold = 0.93
    goal_x = cfg.corridor_length * cfg.goal_threshold

    env = CorridorEnv(cfg, render_mode="human" if not args.headless else None)

    # Initial reset (so the drone and obstacle are spawned)
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.cfg.action_space

    # Optional view delay so you can position the camera
    if (not args.headless) and args.view_delay > 0.0:
        print(f"\n⏳ Waiting {args.view_delay:.1f}s so you can adjust the camera...\n")
        time.sleep(args.view_delay)

    # ---------------------------------
    # Load agent
    # ---------------------------------
    agent = DreamerAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        latent_dim=128,
        hidden_dim=256,
        buffer_size=1000,
        gamma=0.99,
        horizon=10,
    )
    agent.load(args.checkpoint)

    # ---------------------------------
    # Pretty header
    # ---------------------------------
    print(f"\n{'='*60}")
    print(f"🚁 WATCHING SINGLE DRONE")
    print(f"{'='*60}")
    print(f"Corridor:  {cfg.corridor_length}m")

    if cfg.obstacle_enabled:
        obst_desc = f"YES at x={cfg.obstacle_x:.1f}m, y={cfg.obstacle_y:.1f}m, size={cfg.obstacle_size:.2f}m"
        if args.random_obstacle:
            obst_desc += " (randomized)"
    else:
        obst_desc = "NO"

    print(f"Goal:      x ≥ {goal_x:.1f}m")
    print(f"Obstacle:  {obst_desc}")
    print(f"{'='*60}")
    print(f"\nStarting drone navigation...\n")

    # ---------------------------------
    # Rollout
    # ---------------------------------
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    max_x = 0.0
    step = 0
    total_reward = 0.0

    # Viewer success tolerance: treat 99% of goal distance as success
    success_pos_threshold = 0.99 * goal_x

    print(f"{'Step':>5} | {'X':>6} | {'Y':>6} | {'Z':>5} | {'Action (fwd, lat, yaw)':>25} | {'Reward':>8}")
    print(f"{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*25}-+-{'-'*8}")

    done = False
    while not done:
        with torch.no_grad():
            action = agent.act(obs, explore=False)

        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]
        done = terminated.item() or truncated.item()

        pos = env._robot.data.root_pos_w[0]
        x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
        max_x = max(max_x, x)
        total_reward += reward.item()
        step += 1

        if step % 10 == 0 or done:
            act_str = f"({action[0,0]:.2f}, {action[0,1]:.2f}, {action[0,2]:.2f})"
            print(f"{step:5d} | {x:6.2f} | {y:6.2f} | {z:5.2f} | {act_str:>25} | {reward.item():8.2f}")

        if args.slow:
            time.sleep(0.05)

    # ---------------------------------
    # Result summary
    # ---------------------------------
    print(f"\n{'='*60}")

    # Viewer-level success: accept 99% of corridor as success
    reached_goal_view = max_x >= success_pos_threshold

    if reached_goal_view:
        print(
            f"✅ SUCCESS! Drone reached {max_x:.2f}m "
            f"(viewer threshold {success_pos_threshold:.2f}m, env goal {goal_x:.2f}m)"
        )
    else:
        stats = env.get_statistics()
        cb = stats.get("crash_breakdown", {})
        if cb.get("wall", 0) > 0:
            crash_type = "WALL"
        elif cb.get("obstacle", 0) > 0:
            crash_type = "OBSTACLE"
        elif cb.get("ground", 0) > 0:
            crash_type = "GROUND"
        elif cb.get("ceiling", 0) > 0:
            crash_type = "CEILING"
        else:
            crash_type = "TIMEOUT"

        print(f"❌ FAILED: {crash_type} at x={max_x:.2f}m (env goal {goal_x:.2f}m)")

    print(f"\n📊 Summary:")
    print(f"   Max distance: {max_x:.2f}m / {goal_x:.2f}m ({100.0 * max_x / goal_x:.1f}%)")
    print(f"   Total steps:  {step}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"{'='*60}\n")

    simulation_app.close()


if __name__ == "__main__":
    watch()
