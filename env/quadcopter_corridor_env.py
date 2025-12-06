"""
QUADCOPTER CORRIDOR ENVIRONMENT - CLEAN VERSION
================================================

Goal: Fly down a corridor, avoid obstacle, reach the end.

Inputs (Observations - 47 dims):
    - 32 lidar rays: normalized distance to walls/obstacles (0=touching, 1=max range)
    - 3 body velocities: vx, vy, vz (normalized by max_velocity)
    - 3 angular velocities: roll rate, pitch rate, yaw rate
    - 3 orientation: roll, pitch, yaw angles
    - 3 position: x_progress (0-1), y_centered (-1 to 1), z_altitude
    - 2 lidar summary: min_distance, mean_distance
    - 1 time_remaining: fraction of episode left

Outputs (Actions - 3 dims):
    - action[0]: forward velocity command  (-1 to +1) -> (-1.5 to +1.5 m/s)
    - action[1]: lateral velocity command  (-1 to +1) -> (-0.5 to +0.5 m/s)  
    - action[2]: yaw rate command          (-1 to +1) -> (-0.5 to +0.5 rad/s)

Rewards:
    - Forward progress: reward for moving toward goal
    - Crash penalty: -50 for hitting walls/ground/obstacle
    - Success bonus: +100 for reaching goal
    - Small time penalty: encourages efficiency

Episode ends when:
    - SUCCESS: Drone reaches 95% of corridor length
    - CRASH: Drone hits wall, ground, ceiling, or obstacle
    - TIMEOUT: Episode time limit reached
"""

from __future__ import annotations
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


def get_crazyflie_cfg():
    """Crazyflie quadcopter asset configuration."""
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={".*": 0.0},
            joint_vel={"m1_joint": 0.0, "m2_joint": 0.0, "m3_joint": 0.0, "m4_joint": 0.0},
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
        },
    )


@configclass
class CorridorEnvCfg(DirectRLEnvCfg):
    """Configuration for the corridor navigation environment."""
    
    # Episode settings
    episode_length_s: float = 20.0  # 20 second episodes
    decimation: int = 2
    
    # Observation and action dimensions
    observation_space: int = 47
    action_space: int = 3
    
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1/100, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=12.0,
        replicate_physics=True,
    )
    robot: ArticulationCfg = None
    
    # Corridor geometry
    corridor_length: float = 10.0  # meters
    corridor_width: float = 4.0    # meters
    corridor_height: float = 2.0   # meters
    
    # Drone settings
    target_altitude: float = 0.5   # meters
    max_velocity: float = 2.0      # m/s
    drone_radius: float = 0.15     # meters (for collision)
    drone_height: float = 0.10     # meters
    
    # Control gains
    kp_z: float = 25.0   # Altitude P gain
    kd_z: float = 8.0    # Altitude D gain
    max_horiz_force: float = 0.15  # N
    max_yaw_torque: float = 0.002  # N·m
    
    # Lidar settings
    num_lidar_rays: int = 32
    lidar_fov_degrees: float = 120.0
    lidar_max_range: float = 5.0
    
    # Obstacle settings
    obstacle_enabled: bool = True
    obstacle_x: float = 6.0        # meters from start
    obstacle_y: float = 0.0        # lateral position (0 = center)
    obstacle_size: float = 0.8     # meters (width/depth of obstacle)
    
    # Success threshold
    goal_threshold: float = 0.95   # Reach 95% of corridor to succeed


class CorridorEnv(DirectRLEnv):
    """
    Quadcopter corridor navigation environment.
    
    The drone must fly from start (x=0) to end (x=corridor_length),
    avoiding walls and an obstacle in the middle.
    """
    
    cfg: CorridorEnvCfg
    
    def __init__(self, cfg: CorridorEnvCfg, render_mode="human", **kwargs):
        if cfg.robot is None:
            cfg.robot = get_crazyflie_cfg().replace(prim_path="/World/envs/env_.*/Robot")
        
        super().__init__(cfg, render_mode, **kwargs)
        
        device = self.device
        n = self.num_envs
        
        # Force/torque buffers for control
        self._thrust = torch.zeros(n, 1, 3, device=device)
        self._moment = torch.zeros(n, 1, 3, device=device)
        self._body_id = self._robot.find_bodies("body")[0]
        
        # Physical properties
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity = torch.tensor(self.sim.cfg.gravity, device=device).norm()
        self._hover_thrust = (self._robot_mass * self._gravity).item()
        
        # Episode tracking - PER ENVIRONMENT
        self._episode_max_x = torch.zeros(n, device=device)      # Max x reached this episode
        self._episode_start_x = torch.zeros(n, device=device)    # Starting x (always 0)
        self._prev_x = torch.zeros(n, device=device)             # Previous step x for delta
        
        # Statistics tracking - CUMULATIVE across all episodes
        self._stats_episodes_completed = 0
        self._stats_successes = 0
        self._stats_crashes = 0
        self._stats_timeouts = 0
        self._stats_total_distance = 0.0  # Sum of max distances from completed episodes
        self._stats_crash_types = {"ground": 0, "ceiling": 0, "wall": 0, "obstacle": 0}
        
        # Lidar angles (precomputed)
        fov = math.radians(cfg.lidar_fov_degrees)
        self._lidar_angles = torch.linspace(-fov/2, fov/2, cfg.num_lidar_rays, device=device)
        
        # Goal position
        self._goal_x = cfg.corridor_length * cfg.goal_threshold
        
        self._print_env_info()
    
    def _print_env_info(self):
        """Print environment configuration on startup."""
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("CORRIDOR NAVIGATION ENVIRONMENT")
        print("=" * 60)
        print(f"Corridor: {cfg.corridor_length}m long × {cfg.corridor_width}m wide")
        print(f"Goal: Reach x = {self._goal_x:.1f}m ({cfg.goal_threshold*100:.0f}% of corridor)")
        print(f"Obstacle: {'ENABLED' if cfg.obstacle_enabled else 'DISABLED'}", end="")
        if cfg.obstacle_enabled:
            print(f" at x={cfg.obstacle_x}m, size={cfg.obstacle_size}m")
        else:
            print()
        print(f"Drone: mass={self._robot_mass:.4f}kg, hover_thrust={self._hover_thrust:.4f}N")
        print(f"Episode: {cfg.episode_length_s}s max ({self.max_episode_length} steps)")
        print(f"Environments: {self.num_envs} parallel")
        print("=" * 60 + "\n")
    
    # ==================== SCENE SETUP ====================
    
    def _setup_scene(self):
        """Create the corridor environment with walls and obstacle."""
        cfg = self.cfg
        
        # Robot
        self._robot = Articulation(cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        
        # Corridor walls
        wall_length = cfg.corridor_length + 2.0
        half_width = cfg.corridor_width / 2
        
        wall_cfg = sim_utils.CuboidCfg(
            size=(wall_length, 0.1, cfg.corridor_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
        )
        
        # Left wall (negative Y)
        wall_cfg.func(
            "/World/envs/env_./WallLeft",
            wall_cfg,
            translation=(cfg.corridor_length/2, -half_width - 0.05, cfg.corridor_height/2),
        )
        
        # Right wall (positive Y)
        wall_cfg.func(
            "/World/envs/env_./WallRight", 
            wall_cfg,
            translation=(cfg.corridor_length/2, half_width + 0.05, cfg.corridor_height/2),
        )
        
        # Obstacle (if enabled)
        if cfg.obstacle_enabled:
            obs_cfg = sim_utils.CuboidCfg(
                size=(cfg.obstacle_size, cfg.obstacle_size, cfg.corridor_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            )
            obs_cfg.func(
                "/World/envs/env_./Obstacle",
                obs_cfg,
                translation=(cfg.obstacle_x, cfg.obstacle_y, cfg.corridor_height/2),
            )
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/light", light_cfg)
    
    # ==================== CONTROL ====================
    
    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Normalize
        norm = torch.sqrt(w*w + x*x + y*y + z*z + 1e-8)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        R = torch.zeros(quat.shape[0], 3, 3, device=quat.device)
        
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        
        return R
    
    def _get_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle from quaternion."""
        return torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2]**2 + quat[:, 3]**2),
        )
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Convert velocity commands to body forces.
        
        Actions (all normalized -1 to +1):
            [0] forward velocity  -> -1.5 to +1.5 m/s
            [1] lateral velocity  -> -0.5 to +0.5 m/s
            [2] yaw rate          -> -0.5 to +0.5 rad/s
        """
        actions = actions.clamp(-1.0, 1.0)
        
        pos_w = self._robot.data.root_pos_w
        vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        quat_w = self._robot.data.root_quat_w
        
        # Rotation matrix: body -> world
        R = self._quat_to_rotation_matrix(quat_w)
        
        # === Altitude control (PD controller) ===
        z_error = self.cfg.target_altitude - pos_w[:, 2]
        vz_error = -vel_w[:, 2]
        
        thrust_z = self._hover_thrust + self.cfg.kp_z * z_error + self.cfg.kd_z * vz_error
        thrust_z = thrust_z.clamp(0.5 * self._hover_thrust, 1.5 * self._hover_thrust)
        
        # === Horizontal velocity control ===
        vx_desired = actions[:, 0] * 1.5  # m/s forward
        vy_desired = actions[:, 1] * 0.5  # m/s lateral
        
        vx_error = vx_desired - vel_w[:, 0]
        vy_error = vy_desired - vel_w[:, 1]
        
        # Simple proportional control with force limits
        max_accel = 4.0  # m/s^2
        Fx = (self._robot_mass * max_accel * vx_error.clamp(-1, 1)).clamp(
            -self.cfg.max_horiz_force, self.cfg.max_horiz_force
        )
        Fy = (self._robot_mass * max_accel * vy_error.clamp(-1, 1) * 0.5).clamp(
            -self.cfg.max_horiz_force * 0.5, self.cfg.max_horiz_force * 0.5
        )
        
        # Transform world forces to body frame
        F_world = torch.stack([Fx, Fy, thrust_z], dim=-1).unsqueeze(-1)
        F_body = torch.matmul(R.transpose(1, 2), F_world).squeeze(-1)
        
        # === Yaw control ===
        yaw_rate_desired = actions[:, 2] * 0.5  # rad/s
        yaw_error = yaw_rate_desired - ang_vel_b[:, 2]
        yaw_torque = yaw_error * self.cfg.max_yaw_torque
        
        # Store forces for application
        self._thrust[:, 0, :] = F_body
        self._moment[:, 0, 2] = yaw_torque
    
    def _apply_action(self):
        """Apply computed forces to the robot."""
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )
    
    # ==================== OBSERVATIONS ====================
    
    def _compute_lidar(self) -> torch.Tensor:
        """
        Compute lidar readings for each environment.
        Returns normalized distances (0 = collision, 1 = max range).
        """
        n = self.num_envs
        cfg = self.cfg
        max_range = cfg.lidar_max_range
        half_w = cfg.corridor_width / 2
        
        pos = self._robot.data.root_pos_w
        yaw = self._get_yaw(self._robot.data.root_quat_w)
        
        # Ray directions in world frame
        world_angles = self._lidar_angles.unsqueeze(0) + yaw.unsqueeze(1)
        ray_dx = torch.cos(world_angles)
        ray_dy = torch.sin(world_angles)
        
        x = pos[:, 0].unsqueeze(1)
        y = pos[:, 1].unsqueeze(1)
        
        distances = torch.full((n, cfg.num_lidar_rays), max_range, device=self.device)
        
        # Check wall intersections
        # Left wall (y = -half_w)
        t_left = (-half_w - y) / (ray_dy + 1e-8)
        valid = (t_left > 0) & (t_left < max_range) & (ray_dy.abs() > 1e-6)
        hit_x = x + t_left * ray_dx
        valid = valid & (hit_x >= 0) & (hit_x <= cfg.corridor_length)
        distances = torch.where(valid, torch.minimum(distances, t_left), distances)
        
        # Right wall (y = +half_w)
        t_right = (half_w - y) / (ray_dy + 1e-8)
        valid = (t_right > 0) & (t_right < max_range) & (ray_dy.abs() > 1e-6)
        hit_x = x + t_right * ray_dx
        valid = valid & (hit_x >= 0) & (hit_x <= cfg.corridor_length)
        distances = torch.where(valid, torch.minimum(distances, t_right), distances)
        
        # Check obstacle intersection (if enabled)
        if cfg.obstacle_enabled:
            obs_x = cfg.obstacle_x
            obs_y = cfg.obstacle_y
            obs_radius = cfg.obstacle_size / 2 * 1.414  # Diagonal for square obstacle
            
            # Vector to obstacle center
            to_obs_x = obs_x - x
            to_obs_y = obs_y - y
            
            # Project onto ray
            t_obs = to_obs_x * ray_dx + to_obs_y * ray_dy
            
            # Closest point on ray to obstacle
            closest_x = x + t_obs * ray_dx
            closest_y = y + t_obs * ray_dy
            perp_dist = torch.sqrt((closest_x - obs_x)**2 + (closest_y - obs_y)**2)
            
            # Check if ray hits obstacle
            valid = (t_obs > 0) & (t_obs < max_range) & (perp_dist < obs_radius)
            distances = torch.where(valid, torch.minimum(distances, t_obs), distances)
        
        return (distances / max_range).clamp(0, 1)
    
    def _get_observations(self) -> dict:
        """Compute observations for all environments."""
        cfg = self.cfg
        
        lidar = self._compute_lidar()
        
        pos = self._robot.data.root_pos_w
        vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        quat = self._robot.data.root_quat_w
        gravity_b = self._robot.data.projected_gravity_b
        
        # Normalize velocities
        vx = (vel_b[:, 0] / cfg.max_velocity).clamp(-1, 1)
        vy = (vel_b[:, 1] / cfg.max_velocity).clamp(-1, 1)
        vz = (vel_b[:, 2] / cfg.max_velocity).clamp(-1, 1)
        
        # Normalize angular velocities
        omega_x = (ang_vel_b[:, 0] / 2.0).clamp(-1, 1)
        omega_y = (ang_vel_b[:, 1] / 2.0).clamp(-1, 1)
        omega_z = (ang_vel_b[:, 2] / 2.0).clamp(-1, 1)
        
        # Orientation from gravity projection
        roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2]) / math.pi
        pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2]) / math.pi
        yaw = self._get_yaw(quat) / math.pi
        
        # Position (normalized)
        x_progress = (pos[:, 0] / cfg.corridor_length).clamp(0, 1)
        y_centered = (pos[:, 1] / (cfg.corridor_width / 2)).clamp(-1, 1)
        z_altitude = (pos[:, 2] / cfg.corridor_height).clamp(0, 1)
        
        # Lidar summary
        min_dist = lidar.min(dim=1)[0]
        mean_dist = lidar.mean(dim=1)
        
        # Time remaining
        time_left = (self.max_episode_length - self.episode_length_buf) / self.max_episode_length
        
        obs = torch.cat([
            lidar,                          # 32
            vx.unsqueeze(-1),               # 1
            vy.unsqueeze(-1),               # 1
            vz.unsqueeze(-1),               # 1
            omega_x.unsqueeze(-1),          # 1
            omega_y.unsqueeze(-1),          # 1
            omega_z.unsqueeze(-1),          # 1
            roll.unsqueeze(-1),             # 1
            pitch.unsqueeze(-1),            # 1
            yaw.unsqueeze(-1),              # 1
            x_progress.unsqueeze(-1),       # 1
            y_centered.unsqueeze(-1),       # 1
            z_altitude.unsqueeze(-1),       # 1
            min_dist.unsqueeze(-1),         # 1
            mean_dist.unsqueeze(-1),        # 1
            time_left.unsqueeze(-1),        # 1
        ], dim=-1)  # Total: 47
        
        return {"policy": obs}
    
    # ==================== COLLISION DETECTION ====================
    
    def _check_collisions(self) -> tuple[torch.Tensor, dict]:
        """
        Check for collisions in all environments.
        
        Returns:
            crashed: Boolean tensor [num_envs] - True if crashed
            crash_info: Dict with breakdown by type
        """
        cfg = self.cfg
        pos = self._robot.data.root_pos_w
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        
        half_w = cfg.corridor_width / 2
        
        # Check each collision type
        hit_ground = z < 0.08
        hit_ceiling = z > cfg.corridor_height - 0.1
        hit_wall = y.abs() > (half_w - cfg.drone_radius)
        
        # Obstacle collision
        hit_obstacle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if cfg.obstacle_enabled:
            dx = (x - cfg.obstacle_x).abs()
            dy = (y - cfg.obstacle_y).abs()
            half_obs = cfg.obstacle_size / 2 + cfg.drone_radius
            hit_obstacle = (dx < half_obs) & (dy < half_obs) & (z < cfg.corridor_height)
        
        crashed = hit_ground | hit_ceiling | hit_wall | hit_obstacle
        
        crash_info = {
            "ground": hit_ground,
            "ceiling": hit_ceiling,
            "wall": hit_wall,
            "obstacle": hit_obstacle,
        }
        
        return crashed, crash_info
    
    # ==================== REWARDS ====================
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards for all environments.
        
        Reward structure:
            - Forward progress: +10 * delta_x (reward moving toward goal)
            - Backward penalty: -20 * delta_x (penalize going backward)
            - Small time penalty: -0.1 per step (encourage efficiency)
            - Crash penalty: -50
            - Success bonus: +100
        """
        pos = self._robot.data.root_pos_w
        x = pos[:, 0]
        
        # Track max x reached
        self._episode_max_x = torch.maximum(self._episode_max_x, x)
        
        # Forward progress reward
        delta_x = x - self._prev_x
        self._prev_x = x.clone()
        
        # Reward for progress, penalty for going backward
        reward = torch.where(
            delta_x > 0,
            delta_x * 10.0,    # Reward forward movement
            delta_x * 20.0,    # Penalize backward movement (negative * positive = negative)
        )
        
        # Time penalty (small, to encourage finishing quickly)
        reward = reward - 0.1
        
        # Crash penalty
        crashed, _ = self._check_collisions()
        reward = reward - crashed.float() * 50.0
        
        # Success bonus
        reached_goal = x >= self._goal_x
        reward = reward + reached_goal.float() * 100.0
        
        return reward
    
    # ==================== TERMINATION ====================
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which episodes are done.
        
        Returns:
            terminated: True if episode ended due to crash
            truncated: True if episode ended due to timeout or success
        """
        pos = self._robot.data.root_pos_w
        x = pos[:, 0]
        
        # Check termination conditions
        crashed, crash_info = self._check_collisions()
        reached_goal = x >= self._goal_x
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Update statistics for completed episodes
        done = crashed | reached_goal | timed_out
        
        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            
            for idx in done_indices:
                i = idx.item()
                max_dist = self._episode_max_x[i].item()
                self._stats_episodes_completed += 1
                self._stats_total_distance += max_dist
                
                if crashed[i]:
                    self._stats_crashes += 1
                    # Track crash type
                    for ctype, mask in crash_info.items():
                        if mask[i]:
                            self._stats_crash_types[ctype] += 1
                            break
                elif reached_goal[i]:
                    self._stats_successes += 1
                else:  # timed_out
                    self._stats_timeouts += 1
        
        # terminated = crash, truncated = timeout or success
        return crashed, (timed_out | reached_goal)
    
    # ==================== RESET ====================
    
    def _reset_idx(self, env_ids=None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        n = len(env_ids)
        self._robot.reset(env_ids)
        
        # Set initial state
        state = self._robot.data.default_root_state[env_ids].clone()
        state[:, 0] = 0.0  # x = 0
        state[:, 1] = (torch.rand(n, device=self.device) - 0.5) * 0.5  # Small random y
        state[:, 2] = self.cfg.target_altitude  # z = target altitude
        
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset episode tracking
        self._episode_max_x[env_ids] = 0.0
        self._prev_x[env_ids] = 0.0
        
        super()._reset_idx(env_ids)
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> dict:
        """
        Get training statistics.
        
        Returns dict with:
            - episodes: Total episodes completed
            - success_rate: Percentage reaching goal
            - crash_rate: Percentage crashing
            - timeout_rate: Percentage timing out
            - avg_distance: Average max distance reached
            - crash_breakdown: Dict of crashes by type
        """
        total = max(self._stats_episodes_completed, 1)
        
        return {
            "episodes": self._stats_episodes_completed,
            "successes": self._stats_successes,
            "crashes": self._stats_crashes,
            "timeouts": self._stats_timeouts,
            "success_rate": 100.0 * self._stats_successes / total,
            "crash_rate": 100.0 * self._stats_crashes / total,
            "timeout_rate": 100.0 * self._stats_timeouts / total,
            "avg_distance": self._stats_total_distance / total,
            "goal_distance": self._goal_x,
            "crash_breakdown": self._stats_crash_types.copy(),
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self._stats_episodes_completed = 0
        self._stats_successes = 0
        self._stats_crashes = 0
        self._stats_timeouts = 0
        self._stats_total_distance = 0.0
        self._stats_crash_types = {"ground": 0, "ceiling": 0, "wall": 0, "obstacle": 0}
