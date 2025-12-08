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
parser.add_argument("--checkpoint", type=str, default=None)
args = parser.parse_args()
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
import torch
import numpy as np

try:
    from isaaclab_tasks.direct.quadcopter.quadcopter_corridor_env import CorridorEnv, CorridorEnvCfg
    from isaaclab_tasks.direct.quadcopter.minimal_dreamer import SimpleDreamer
except ImportError:
    from quadcopter_corridor_env import CorridorEnv, CorridorEnvCfg
    from simple_dreamerv34 import MinimalDreamer
def train():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
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
    
    start_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        start_step = agent.train_steps
        print(f"Resumed from checkpoint: {args.checkpoint}, step {start_step}")    
    checkpoint_dir = "checkpoints_dreamer"
    os.makedirs(checkpoint_dir, exist_ok=True)    
    global_step = start_step
    best_success_rate = 0.0
    best_distance = 0.0
    episode_rewards = deque(maxlen=100)
    current_reward = torch.zeros(args.num_envs, device=device)    
    log_data = {
        'steps': [],
        'success_rate': [],
        'crash_rate': [],
        'avg_distance': [],
        'model_loss': [],
        'actor_loss': [],
        'critic_loss': [],
    }    
    print(f"Starting training | Steps: {args.total_steps:,} | Envs: {args.num_envs} | Corridor: {cfg.corridor_length}m")    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]    
    prefill_steps = 2000
    train_every = 4
    log_every = 2000
    save_every = 20000    
    training_start = time.time()
    losses = {}    
    while global_step < args.total_steps:
        
        if global_step < prefill_steps:
            action = torch.rand(args.num_envs, action_dim, device=device) * 2 - 1
        else:
            action = agent.act(obs, explore=True)
        
        obs_dict, reward, terminated, truncated, info = env.step(action.clamp(-1, 1))
        next_obs = obs_dict["policy"]
        done = terminated | truncated        
        agent.store(obs, action, reward, next_obs, done)        
        current_reward += reward
        for i in range(args.num_envs):
            if done[i]:
                episode_rewards.append(current_reward[i].item())
                current_reward[i] = 0
        
        obs = next_obs
        global_step += args.num_envs
        
        if global_step >= prefill_steps and global_step % train_every == 0:
            try:
                losses = agent.train_step(batch_size=64)
            except Exception as e:
                print(f"Training error at step {global_step}: {e}")
                losses = {}        
        if global_step % log_every < args.num_envs and global_step > 0:
            stats = env.get_statistics()
            elapsed = time.time() - training_start
            steps_per_sec = global_step / max(elapsed, 1)            
            if stats['success_rate'] > best_success_rate:
                best_success_rate = stats['success_rate']
                best_path = os.path.join(checkpoint_dir, 'best.pt')
                agent.save(best_path)            
            if stats['avg_distance'] > best_distance:
                best_distance = stats['avg_distance']            
            log_data['steps'].append(global_step)
            log_data['success_rate'].append(stats['success_rate'])
            log_data['crash_rate'].append(stats['crash_rate'])
            log_data['avg_distance'].append(stats['avg_distance'])
            if losses:
                log_data['model_loss'].append(losses.get('model_loss', 0))
                log_data['actor_loss'].append(losses.get('actor_loss', 0))
                log_data['critic_loss'].append(losses.get('critic_loss', 0))            
            print(f"Step {global_step:,}/{args.total_steps:,} | {elapsed/60:.1f}min | Success: {stats['success_rate']:.1f}% | Distance: {stats['avg_distance']:.2f}m | {steps_per_sec:.0f} steps/s")            
            env.reset_statistics()
        
        if global_step % save_every < args.num_envs and global_step > 0:
            path = os.path.join(checkpoint_dir, f'step_{global_step}.pt')
            agent.save(path)            
            log_path = os.path.join(checkpoint_dir, 'training_log.pt')
            torch.save(log_data, log_path)    
    total_time = time.time() - training_start    
    print(f"Training complete | Time: {total_time/60:.1f}min | Best success: {best_success_rate:.1f}% | Best distance: {best_distance:.2f}m")   
    final_path = os.path.join(checkpoint_dir, 'final.pt')
    agent.save(final_path)    
    log_path = os.path.join(checkpoint_dir, 'training_log.pt')
    torch.save(log_data, log_path)    
    simulation_app.close()


if __name__ == "__main__":
    train()
