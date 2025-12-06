"""
PPO TRAINING SCRIPT - CLEAN VERSION WITH EXTENSIVE LOGGING
===========================================================

This script trains a quadcopter to navigate down a corridor and avoid obstacles.

What you'll see in the logs:
    - How many drones succeeded, crashed, or timed out
    - Where crashes are happening (ground, wall, obstacle)
    - Average distance traveled
    - Whether the policy is actually improving

Run with:
    python train_corridor_ppo.py --num_envs 32 --headless
    
Watch training with GUI:
    python train_corridor_ppo.py --num_envs 4
"""

import argparse
import os
import time
from datetime import datetime

# Parse args BEFORE Isaac imports
parser = argparse.ArgumentParser(description="Train PPO for corridor navigation")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
args = parser.parse_args()

# Isaac imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# Import our environment
# Import our environment from isaaclab_tasks package
from isaaclab_tasks.direct.quadcopter.quadcopter_corridor_env import (
    CorridorEnv,
    CorridorEnvCfg,
)



# ============================================================================
# NEURAL NETWORK
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Actor-Critic neural network for PPO.
    
    Actor: Outputs mean of action distribution
    Critic: Outputs value estimate
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        # Actor network (outputs action mean)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        
        # Critic network (outputs value)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # Learnable log standard deviation
        # Initialize slightly negative for reasonable exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
    
    def forward(self, obs: torch.Tensor):
        """Forward pass returning action mean, std, and value."""
        action_mean = self.actor(obs)
        # Clamp log_std to prevent too much or too little exploration
        action_std = torch.exp(self.log_std.clamp(-2.0, 0.5))  # std in [0.135, 1.65]
        value = self.critic(obs)
        return action_mean, action_std, value
    
    def get_action(self, obs: torch.Tensor):
        """Sample action from policy distribution."""
        action_mean, action_std, value = self(obs)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value
    
    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Evaluate given actions under current policy."""
        action_mean, action_std, value = self(obs)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy
    
    def act_deterministic(self, obs: torch.Tensor):
        """Get deterministic action (mean) for evaluation."""
        action_mean, _, value = self(obs)
        return action_mean, value


# ============================================================================
# ROLLOUT BUFFER
# ============================================================================

class RolloutBuffer:
    """Stores rollout data for PPO updates."""
    
    def __init__(self, horizon: int, num_envs: int, obs_dim: int, action_dim: int, device):
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device
        
        self.observations = torch.zeros((horizon, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((horizon, num_envs, action_dim), device=device)
        self.log_probs = torch.zeros((horizon, num_envs), device=device)
        self.rewards = torch.zeros((horizon, num_envs), device=device)
        self.values = torch.zeros((horizon, num_envs), device=device)
        self.dones = torch.zeros((horizon, num_envs), dtype=torch.bool, device=device)
        
        self.ptr = 0
    
    def add(self, obs, action, log_prob, reward, value, done):
        """Add one timestep of data."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.horizon
    
    def compute_returns(self, next_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute returns and advantages using GAE."""
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)
        
        last_gae = 0
        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            not_done = (~self.dones[t]).float()
            delta = self.rewards[t] + gamma * next_val * not_done - self.values[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]
        
        return returns, advantages
    
    def get_batches(self, returns, advantages, batch_size):
        """Yield minibatches for PPO updates."""
        total_size = self.horizon * self.num_envs
        indices = torch.randperm(total_size, device=self.device)
        
        # Flatten data
        obs_flat = self.observations.reshape(total_size, -1)
        actions_flat = self.actions.reshape(total_size, -1)
        log_probs_flat = self.log_probs.reshape(total_size)
        returns_flat = returns.reshape(total_size)
        advantages_flat = advantages.reshape(total_size)
        
        for start in range(0, total_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            
            yield (
                obs_flat[idx],
                actions_flat[idx],
                log_probs_flat[idx],
                returns_flat[idx],
                advantages_flat[idx],
            )


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train(
    env: CorridorEnv,
    policy: PolicyNetwork,
    num_updates: int = 500,
    horizon: int = 256,
    batch_size: int = 512,
    num_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    checkpoint_dir: str = "checkpoints",
):
    """
    Train PPO policy.
    
    Args:
        env: The corridor environment
        policy: The policy network
        num_updates: Number of PPO updates to perform
        horizon: Steps per rollout
        batch_size: Minibatch size for updates
        num_epochs: PPO epochs per update
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_eps: PPO clip epsilon
        vf_coef: Value loss coefficient
        ent_coef: Entropy bonus coefficient
        max_grad_norm: Gradient clipping threshold
        checkpoint_dir: Where to save checkpoints
    """
    
    device = env.device
    obs_dim = env.cfg.observation_space
    action_dim = env.cfg.action_space
    
    # Setup
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer(horizon, env.num_envs, obs_dim, action_dim, device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tracking
    best_success_rate = 0.0
    best_avg_distance = 0.0
    training_start = time.time()
    
    # Print training configuration
    print("\n" + "=" * 70)
    print("PPO TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Updates:         {num_updates}")
    print(f"Horizon:         {horizon} steps")
    print(f"Batch size:      {batch_size}")
    print(f"Epochs:          {num_epochs}")
    print(f"Learning rate:   {lr}")
    print(f"Gamma:           {gamma}")
    print(f"GAE lambda:      {gae_lambda}")
    print(f"Clip epsilon:    {clip_eps}")
    print(f"Entropy coef:    {ent_coef}")
    print(f"Samples/update:  {horizon * env.num_envs}")
    print("=" * 70 + "\n")
    
    # Initial reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # ==================== TRAINING LOOP ====================
    
    for update in range(1, num_updates + 1):
        update_start = time.time()
        
        # Reset statistics for this update
        env.reset_statistics()
        
        # ============ COLLECT ROLLOUT ============
        policy.eval()
        
        for step in range(horizon):
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs)
            
            # Clamp actions to valid range
            action_clamped = action.clamp(-1.0, 1.0)
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action_clamped)
            next_obs = obs_dict["policy"]
            done = terminated | truncated
            
            # Store in buffer
            buffer.add(obs, action, log_prob, reward, value.squeeze(-1), done)
            
            obs = next_obs
        
        # Get statistics from this rollout
        stats = env.get_statistics()
        
        # ============ COMPUTE RETURNS ============
        with torch.no_grad():
            _, _, next_value = policy(obs)
        
        returns, advantages = buffer.compute_returns(
            next_value.squeeze(-1), gamma, gae_lambda
        )
        
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # ============ PPO UPDATE ============
        policy.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch in buffer.get_batches(returns, advantages, batch_size):
                b_obs, b_actions, b_old_log_probs, b_returns, b_advantages = batch
                
                # Evaluate actions
                log_prob, value, entropy = policy.evaluate(b_obs, b_actions)
                
                # Policy loss (PPO clip objective)
                ratio = torch.exp(log_prob - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (with clipping for stability)
                value_loss = 0.5 * (value.squeeze(-1) - b_returns).pow(2).mean()
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                
                # Track
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_frac += ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                num_batches += 1
        
        # Average losses
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches
        avg_clip_frac = total_clip_frac / num_batches
        
        update_time = time.time() - update_start
        
        # ============ LOGGING ============
        
        # Track best performance
        if stats["success_rate"] > best_success_rate:
            best_success_rate = stats["success_rate"]
        if stats["avg_distance"] > best_avg_distance:
            best_avg_distance = stats["avg_distance"]
        
        # Get action statistics
        action_std = torch.exp(policy.log_std.clamp(-2.0, 0.5))
        
        # Print update summary
        print("\n" + "=" * 70)
        print(f"UPDATE {update}/{num_updates}  |  Time: {update_time:.1f}s  |  Total: {(time.time()-training_start)/60:.1f}min")
        print("=" * 70)
        
        # Episode outcomes
        print(f"\n📊 EPISODE OUTCOMES ({stats['episodes']} episodes completed):")
        print(f"   ✅ Success:  {stats['successes']:4d} ({stats['success_rate']:5.1f}%)  - Reached goal at {stats['goal_distance']:.1f}m")
        print(f"   💥 Crash:    {stats['crashes']:4d} ({stats['crash_rate']:5.1f}%)")
        print(f"   ⏰ Timeout:  {stats['timeouts']:4d} ({stats['timeout_rate']:5.1f}%)")
        
        # Crash breakdown
        if stats['crashes'] > 0:
            cb = stats['crash_breakdown']
            print(f"   └─ Crash types: ground={cb['ground']}, wall={cb['wall']}, obstacle={cb['obstacle']}, ceiling={cb['ceiling']}")
        
        # Distance
        print(f"\n📏 DISTANCE:")
        print(f"   Average:  {stats['avg_distance']:.2f}m / {env.cfg.corridor_length:.1f}m ({100*stats['avg_distance']/env.cfg.corridor_length:.1f}%)")
        print(f"   Best avg: {best_avg_distance:.2f}m")
        
        # Learning metrics
        print(f"\n📈 LEARNING:")
        print(f"   Reward/step: {buffer.rewards.mean().item():+.3f}")
        print(f"   Return:      {returns.mean().item():+.2f}")
        print(f"   Value est:   {buffer.values.mean().item():+.2f}")
        
        # Losses
        print(f"\n📉 LOSSES:")
        print(f"   Policy:  {avg_policy_loss:+.4f}  (negative = improving)")
        print(f"   Value:   {avg_value_loss:.4f}  (lower = better predictions)")
        print(f"   Entropy: {avg_entropy:.4f}  (higher = more exploration)")
        
        # Policy behavior
        print(f"\n🎲 POLICY BEHAVIOR:")
        print(f"   Action std:  [{action_std[0].item():.3f}, {action_std[1].item():.3f}, {action_std[2].item():.3f}]")
        print(f"   Clip frac:   {avg_clip_frac:.3f}  (0.05-0.15 is healthy)")
        print(f"   Grad norm:   {grad_norm:.3f}")
        
        # Assessment
        print(f"\n🔍 ASSESSMENT:")
        if stats['success_rate'] > 90:
            print("   ✅ Excellent! Policy is reliably reaching the goal.")
        elif stats['success_rate'] > 50:
            print("   🟡 Good progress. Policy is learning but not yet reliable.")
        elif stats['crash_rate'] > 50:
            print("   🔴 High crash rate. Policy needs more training or reward tuning.")
        elif stats['avg_distance'] < 2.0:
            print("   🔴 Low distance. Policy may not be learning to move forward.")
        else:
            print("   🟡 Training in progress. Watch for improvement over updates.")
        
        # ============ CHECKPOINTING ============
        
        if update % 50 == 0 or stats['success_rate'] > 95:
            checkpoint = {
                'update': update,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_success_rate': best_success_rate,
                'best_avg_distance': best_avg_distance,
                'config': {
                    'corridor_length': env.cfg.corridor_length,
                    'obstacle_enabled': env.cfg.obstacle_enabled,
                    'obstacle_x': env.cfg.obstacle_x,
                }
            }
            path = os.path.join(checkpoint_dir, f'ppo_update_{update}.pt')
            torch.save(checkpoint, path)
            print(f"\n   💾 Checkpoint saved: {path}")
    
    # ==================== TRAINING COMPLETE ====================
    
    total_time = time.time() - training_start
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best success rate: {best_success_rate:.1f}%")
    print(f"Best avg distance: {best_avg_distance:.2f}m")
    print("=" * 70 + "\n")
    
    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, 'ppo_final.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'best_success_rate': best_success_rate,
        'best_avg_distance': best_avg_distance,
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    return policy


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "=" * 70)
    print("QUADCOPTER CORRIDOR NAVIGATION - PPO TRAINING")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {args.seed}")
    print(f"Environments: {args.num_envs}")
    print(f"Headless: {args.headless}")
    print("=" * 70)
    
    # Create environment
    cfg = CorridorEnvCfg()
    cfg.scene.num_envs = args.num_envs
    
    # Environment settings - start simple, then make harder
    cfg.corridor_length = 10.0      # 10 meter corridor
    cfg.obstacle_enabled = True     # Include obstacle
    cfg.obstacle_x = 6.0            # Obstacle at 6m
    cfg.obstacle_y = 0.0            # Obstacle in center
    cfg.obstacle_size = 0.8         # 0.8m obstacle
    cfg.goal_threshold = 0.95       # Must reach 9.5m to succeed
    
    env = CorridorEnv(
        cfg,
        render_mode="human" if not args.headless else None
    )
    
    # Get dimensions
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.cfg.action_space
    
    print(f"\nObservation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Device: {env.device}")
    
    # Create policy
    policy = PolicyNetwork(obs_dim, action_dim, hidden_size=256).to(env.device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=env.device)
        policy.load_state_dict(ckpt['policy_state_dict'])
        print("Checkpoint loaded!")
    
    print(f"\nPolicy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Train
    trained_policy = train(
        env=env,
        policy=policy,
        num_updates=500,
        horizon=256,
        batch_size=512,
        num_epochs=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,  # Lower entropy for more focused learning
        max_grad_norm=0.5,
    )
    
    simulation_app.close()
