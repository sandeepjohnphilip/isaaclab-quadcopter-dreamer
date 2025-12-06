"""
TRAIN MINIMAL DREAMERV2
========================

Simple training script with:
- Extensive logging to understand what's happening
- Checkpoint loading to resume training
- Clear progress indicators

Usage:
    # Fresh training
    cd ~/IsaacLab
    ./isaaclab.sh -p source/standalone/train_minimal_dreamer.py -- --num_envs 8 --headless

    # Resume from checkpoint
    ./isaaclab.sh -p source/standalone/train_minimal_dreamer.py -- --num_envs 8 --headless --checkpoint checkpoints_dreamer/best.pt
"""

import argparse
import os
import time
from datetime import datetime
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_steps", type=int, default=300000)
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint to resume")
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

# Adjust import path as needed
try:
    from isaaclab_tasks.direct.quadcopter.quadcopter_corridor_env import CorridorEnv, CorridorEnvCfg
    from isaaclab_tasks.direct.quadcopter.minimal_dreamer import SimpleDreamer
except ImportError:
    # Try local import
    from quadcopter_corridor_env import CorridorEnv, CorridorEnvCfg
    from simple_dreamerv34  import MinimalDreamer


def train():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==================== ENVIRONMENT ====================
    
    cfg = CorridorEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.corridor_length = 10.0
    cfg.obstacle_enabled = True
    cfg.obstacle_x = 6.0
    cfg.obstacle_y = 0.0
    cfg.obstacle_size = 0.8
    cfg.goal_threshold = 0.93
    
    env = CorridorEnv(cfg, render_mode="human" if not args.headless else None)
    
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.cfg.action_space
    
    # ==================== AGENT ====================
    
    agent = MinimalDreamer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        latent_dim=128,
        hidden_dim=256,
        buffer_capacity=100000,
        imagination_horizon=15,
        gamma=0.99,
        lambda_=0.95,
        model_lr=3e-4,
        actor_lr=1e-4,
        critic_lr=1e-4,
    )
    
    # Load checkpoint if provided
    start_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        start_step = agent.train_steps
        print(f"\n✅ Resumed from checkpoint: {args.checkpoint}")
        print(f"   Starting from step {start_step}")
    
    # ==================== SETUP ====================
    
    checkpoint_dir = "checkpoints_dreamer"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tracking
    global_step = start_step
    best_success_rate = 0.0
    best_distance = 0.0
    episode_rewards = deque(maxlen=100)
    
    # Current episode tracking
    current_reward = torch.zeros(args.num_envs, device=device)
    
    # Training log
    log_data = {
        'steps': [],
        'success_rate': [],
        'crash_rate': [],
        'avg_distance': [],
        'model_loss': [],
        'actor_loss': [],
        'critic_loss': [],
    }
    
    # ==================== PRINT CONFIG ====================
    
    print("\n" + "=" * 70)
    print("MINIMAL DREAMERV2 TRAINING")
    print("=" * 70)
    print(f"Time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total steps:     {args.total_steps:,}")
    print(f"Num envs:        {args.num_envs}")
    print(f"Corridor:        {cfg.corridor_length}m with obstacle at {cfg.obstacle_x}m")
    print(f"Goal:            {cfg.corridor_length * cfg.goal_threshold:.1f}m")
    print(f"\nAgent parameters:")
    print(f"  World model:   {agent.model_params:,}")
    print(f"  Actor:         {agent.actor_params:,}")
    print(f"  Critic:        {agent.critic_params:,}")
    print(f"  Total:         {agent.model_params + agent.actor_params + agent.critic_params:,}")
    print("=" * 70 + "\n")
    
    # ==================== TRAINING LOOP ====================
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    prefill_steps = 2000  # Collect some random data first
    train_every = 4       # Train every N env steps
    log_every = 2000      # Log every N steps
    save_every = 20000    # Save checkpoint every N steps
    
    training_start = time.time()
    losses = {}
    
    while global_step < args.total_steps:
        
        # ============ COLLECT EXPERIENCE ============
        
        # Random exploration during prefill
        if global_step < prefill_steps:
            action = torch.rand(args.num_envs, action_dim, device=device) * 2 - 1
        else:
            action = agent.act(obs, explore=True)
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action.clamp(-1, 1))
        next_obs = obs_dict["policy"]
        done = terminated | truncated
        
        # Store in buffer
        agent.store(obs, action, reward, next_obs, done)
        
        # Track rewards
        current_reward += reward
        for i in range(args.num_envs):
            if done[i]:
                episode_rewards.append(current_reward[i].item())
                current_reward[i] = 0
        
        obs = next_obs
        global_step += args.num_envs
        
        # ============ TRAIN ==============
        
        if global_step >= prefill_steps and global_step % train_every == 0:
            try:
                losses = agent.train_step(batch_size=64)
            except Exception as e:
                print(f"\n❌ Training error at step {global_step}: {e}")
                print("   Continuing without training this step...")
                losses = {}
        
        # ============ LOGGING ============
        
        if global_step % log_every < args.num_envs and global_step > 0:
            stats = env.get_statistics()
            elapsed = time.time() - training_start
            steps_per_sec = global_step / max(elapsed, 1)
            
            # Track best
            if stats['success_rate'] > best_success_rate:
                best_success_rate = stats['success_rate']
                # Save best checkpoint
                best_path = os.path.join(checkpoint_dir, 'best.pt')
                agent.save(best_path)
                print(f"\n🏆 New best! Saved to {best_path}")
            
            if stats['avg_distance'] > best_distance:
                best_distance = stats['avg_distance']
            
            # Store log
            log_data['steps'].append(global_step)
            log_data['success_rate'].append(stats['success_rate'])
            log_data['crash_rate'].append(stats['crash_rate'])
            log_data['avg_distance'].append(stats['avg_distance'])
            if losses:
                log_data['model_loss'].append(losses.get('model_loss', 0))
                log_data['actor_loss'].append(losses.get('actor_loss', 0))
                log_data['critic_loss'].append(losses.get('critic_loss', 0))
            
            # Print status
            print("\n" + "=" * 70)
            print(f"STEP {global_step:,} / {args.total_steps:,}  |  {elapsed/60:.1f}min  |  {steps_per_sec:.0f} steps/s")
            print("=" * 70)
            
            # Phase indicator
            if global_step < prefill_steps:
                print(f"\n📦 PHASE: Prefilling buffer ({global_step}/{prefill_steps})")
            else:
                print(f"\n🧠 PHASE: Training (buffer: {agent.buffer.size:,} samples)")
            
            # Episode outcomes
            print(f"\n📊 EPISODES ({stats['episodes']} completed):")
            print(f"   ✅ Success:  {stats['successes']:4d} ({stats['success_rate']:5.1f}%)  [Best: {best_success_rate:.1f}%]")
            print(f"   💥 Crash:    {stats['crashes']:4d} ({stats['crash_rate']:5.1f}%)")
            print(f"   ⏰ Timeout:  {stats['timeouts']:4d} ({stats['timeout_rate']:5.1f}%)")
            
            if stats['crashes'] > 0:
                cb = stats['crash_breakdown']
                print(f"   └─ ground={cb['ground']}, wall={cb['wall']}, obstacle={cb['obstacle']}, ceiling={cb['ceiling']}")
            
            # Distance
            print(f"\n📏 DISTANCE:")
            print(f"   Average:  {stats['avg_distance']:.2f}m / {cfg.corridor_length:.1f}m ({100*stats['avg_distance']/cfg.corridor_length:.1f}%)")
            print(f"   Best:     {best_distance:.2f}m")
            
            # Rewards
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"\n📈 REWARDS:")
            print(f"   Avg episode reward: {avg_reward:.2f}")
            
            # World model losses
            if losses:
                print(f"\n🌍 WORLD MODEL LOSSES:")
                print(f"   Total:     {losses.get('model_loss', 0):.4f}")
                print(f"   Dynamics:  {losses.get('dynamics_loss', 0):.4f}  (predict next state)")
                print(f"   Recon:     {losses.get('recon_loss', 0):.4f}  (reconstruct obs)")
                print(f"   Reward:    {losses.get('reward_loss', 0):.4f}  (predict reward)")
                print(f"   Done:      {losses.get('done_loss', 0):.4f}  (predict done)")
                
                print(f"\n🎭 ACTOR-CRITIC LOSSES:")
                print(f"   Actor:     {losses.get('actor_loss', 0):.4f}")
                print(f"   Critic:    {losses.get('critic_loss', 0):.4f}")
                
                print(f"\n🔮 IMAGINATION:")
                print(f"   Imagined reward: {losses.get('imagined_reward', 0):.3f}")
                print(f"   Imagined done:   {losses.get('imagined_done', 0):.3f}")
            
            # Assessment
            print(f"\n🔍 ASSESSMENT:")
            if global_step < prefill_steps:
                print("   Collecting initial experience...")
            elif losses.get('dynamics_loss', 1) > 0.1:
                print("   World model still learning environment dynamics...")
            elif stats['success_rate'] > 80:
                print("   ✅ Excellent! Agent reliably reaches the goal.")
            elif stats['success_rate'] > 40:
                print("   🟡 Good progress. Agent is learning.")
            elif stats['crash_rate'] > 50:
                print("   🔴 High crash rate. World model may need more training.")
            elif stats['avg_distance'] < 3:
                print("   🔴 Low distance. Check if world model is learning.")
            else:
                print("   🟡 Training in progress...")
            
            # Reset env stats
            env.reset_statistics()
        
        # ============ SAVE CHECKPOINT ============
        
        if global_step % save_every < args.num_envs and global_step > 0:
            path = os.path.join(checkpoint_dir, f'step_{global_step}.pt')
            agent.save(path)
            
            # Save log
            log_path = os.path.join(checkpoint_dir, 'training_log.pt')
            torch.save(log_data, log_path)
            
            print(f"\n💾 Checkpoint saved: {path}")
    
    # ==================== DONE ====================
    
    total_time = time.time() - training_start
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time:        {total_time/60:.1f} minutes")
    print(f"Total steps:       {global_step:,}")
    print(f"Best success rate: {best_success_rate:.1f}%")
    print(f"Best distance:     {best_distance:.2f}m")
    print("=" * 70)
    
    # Final save
    final_path = os.path.join(checkpoint_dir, 'final.pt')
    agent.save(final_path)
    
    log_path = os.path.join(checkpoint_dir, 'training_log.pt')
    torch.save(log_data, log_path)
    
    print(f"\nFinal model: {final_path}")
    print(f"Best model:  {os.path.join(checkpoint_dir, 'best.pt')}")
    print(f"Training log: {log_path}")
    
    simulation_app.close()


if __name__ == "__main__":
    train()
