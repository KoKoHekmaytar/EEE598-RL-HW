import os
import numpy as np
import json
import pandas as pd

# 🩹 Compatibility patch for deprecated np.int - MUST be before any other imports
if not hasattr(np, "int"):
    np.int = int

# Imports must come AFTER this patch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import torch
import gym  # keep gym, not gymnasium, because dm_control2gym uses gym API
import gymnasium
from utils import set_seed

# Import dm_control2gym after np.int patch
import dm_control2gym


def make_env(seed=0):
    import gym
    import numpy as np

    env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
    
    # ✅ Wrapper to convert old gym API to new gymnasium API for FlattenObservation
    class GymToGymnasiumWrapper(gym.Wrapper):
        """Convert old gym API to gymnasium API (both reset and step)"""
        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple):
                return result
            else:
                return result, {}
        
        def step(self, action):
            result = self.env.step(action)
            # Handle old gym API (4 values) vs new gymnasium API (5 values)
            if len(result) == 5:
                return result
            elif len(result) == 4:
                obs, reward, done, info = result
                terminated = done
                truncated = False
                return obs, reward, terminated, truncated, info
            else:
                raise ValueError(f"Unexpected step() return format: {len(result)} values")
    
    env = GymToGymnasiumWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    
    # ✅ Custom initialization wrapper: Start with pole inverted (θ ≈ π) instead of upright
    class CustomInitWrapper(gym.Wrapper):
        """Wrapper to modify initial state: pole starts inverted (θ ≈ π) instead of upright"""
        def reset(self, **kwargs):
            # Handle old gym API (returns single value) vs new gymnasium API (returns tuple)
            reset_result = self.env.reset(**kwargs)
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            # Navigate to underlying dm_control environment to access physics
            base_env = self.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # Try to access physics model from dm_control2gym structure
            physics = None
            if hasattr(base_env, '_env') and hasattr(base_env._env, 'physics'):
                physics = base_env._env.physics
            elif hasattr(base_env, 'physics'):
                physics = base_env.physics
            elif hasattr(base_env, 'dmcenv') and hasattr(base_env.dmcenv, 'physics'):
                physics = base_env.dmcenv.physics
            
            if physics is not None:
                try:
                    # Set pole angle (theta) to π radians (inverted/downward) with small random variation
                    # This increases difficulty and prevents identical starting conditions
                    pole_angle = np.pi + np.random.uniform(-0.2, 0.2)
                    # For typical dm_control cartpole envs
                    try:
                        physics.named.data.qpos['hinge'] = pole_angle
                    except Exception:
                        # Some wrappers expose qpos as a numpy array, not a named dict
                        if hasattr(physics, "data") and hasattr(physics.data, "qpos"):
                            if len(physics.data.qpos) > 1:
                                physics.data.qpos[1] = pole_angle  # second joint = pole hinge
                            else:
                                physics.data.qpos[:] = pole_angle
                    
                    # Set velocities to zero (start from rest)
                    physics.data.qvel[:] = 0.0
                    
                    # Forward the physics to update state
                    physics.forward()
                    
                    # Debug: Verify physics modification worked
                    if hasattr(physics, "data") and hasattr(physics.data, "qpos"):
                        qpos_after = physics.data.qpos
                        if hasattr(qpos_after, '__len__') and len(qpos_after) > 1:
                            # Print first few resets to verify
                            if not hasattr(self, '_reset_count'):
                                self._reset_count = 0
                            self._reset_count += 1
                            if self._reset_count <= 5:
                                # Try to get the actual values
                                try:
                                    if hasattr(qpos_after, '__getitem__'):
                                        qpos_vals = [qpos_after[i] for i in range(min(2, len(qpos_after)))]
                                    else:
                                        qpos_vals = qpos_after
                                    print(f"[DIAG] Reset {self._reset_count}: qpos after modification = {qpos_vals}")
                                    print(f"       Expected pole angle ≈ π ({np.pi:.3f}), got: {qpos_vals[1] if len(qpos_vals) > 1 else 'N/A'}")
                                except Exception as e:
                                    print(f"[DIAG] Reset {self._reset_count}: Could not read qpos: {e}")
                    
                    # Get updated observation after modifying physics state
                    # ALWAYS manually construct from physics since task.get_observation() returns zeros
                    # Extract state directly from physics: [x, x_dot, theta, theta_dot]
                    qpos = physics.data.qpos
                    qvel = physics.data.qvel
                    
                    # Extract values - handle both array and dict-like structures
                    try:
                        # Try direct indexing first
                        if hasattr(qpos, '__getitem__'):
                            x = float(qpos[0])
                            theta = float(qpos[1])
                        else:
                            x = 0.0
                            theta = float(pole_angle)
                    except (TypeError, ValueError, IndexError, KeyError) as e:
                        # If indexed access fails, use the angle we set
                        x = 0.0
                        theta = float(pole_angle)
                        if self._reset_count <= 3:
                            print(f"[DIAG] qpos indexed access failed: {e}, using pole_angle={pole_angle:.4f}")
                    
                    try:
                        if hasattr(qvel, '__getitem__'):
                            x_dot = float(qvel[0])
                            theta_dot = float(qvel[1])
                        else:
                            x_dot = 0.0
                            theta_dot = 0.0
                    except (TypeError, ValueError, IndexError, KeyError):
                        x_dot = 0.0
                        theta_dot = 0.0
                    
                    # Construct observation array directly from physics state
                    # Preserve the original observation shape (may have 5 dimensions)
                    # Update first 4 elements with physics values, keep the rest from original
                    obs_base = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
                    
                    # Preserve original observation shape and update first 4 elements
                    obs = obs.copy()  # Make a copy to avoid modifying the original
                    obs[:4] = obs_base[:4]  # Update first 4 elements with physics values
                    
                    if self._reset_count <= 3:
                        print(f"[DIAG] Manually constructed obs from physics: [x={x:.4f}, x_dot={x_dot:.4f}, theta={theta:.4f}, theta_dot={theta_dot:.4f}]")
                        print(f"       obs array: {obs}")
                        print(f"       obs shape: {obs.shape}, dtype: {obs.dtype}")
                        print(f"       Original obs shape was: {len(obs) if hasattr(obs, '__len__') else 'N/A'}")
                except Exception as e:
                    # If modification fails, use original observation
                    # Suppress warning for OrderedDict issues - physics modification is optional
                    if "OrderedDict" not in str(e):
                        print(f"Warning: Could not modify initial state: {e}")
            
            # Debug: Verify final observation before returning
            if hasattr(self, '_reset_count') and self._reset_count <= 3:
                print(f"[DIAG] Final obs before return: {obs[:4] if len(obs) >= 4 else obs}")
                if len(obs) >= 4 and np.allclose(obs[:4], 0, atol=1e-6):
                    print(f"       ⚠️  WARNING: Observation is all zeros! Physics modification may not be reflected.")
            
            return obs, info
        
        def step(self, action):
            """Override step() to manually construct observation from physics state"""
            # Call step on wrapped environment
            step_result = self.env.step(action)
            
            # Handle both old gym API (4 values) and new gymnasium API (5 values)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                raise ValueError(f"Unexpected step() return format: {len(step_result)} values")
            
            # Navigate to underlying dm_control environment to access physics
            base_env = self.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # Try to access physics model from dm_control2gym structure
            physics = None
            if hasattr(base_env, '_env') and hasattr(base_env._env, 'physics'):
                physics = base_env._env.physics
            elif hasattr(base_env, 'physics'):
                physics = base_env.physics
            elif hasattr(base_env, 'dmcenv') and hasattr(base_env.dmcenv, 'physics'):
                physics = base_env.dmcenv.physics
            
            # Manually construct observation from physics state (same as in reset())
            # This is necessary because task.get_observation() returns zeros
            if physics is not None:
                try:
                    # Ensure physics state is up to date
                    physics.forward()
                    
                    qpos = physics.data.qpos
                    qvel = physics.data.qvel
                    
                    # Extract values - handle both array and dict-like structures
                    try:
                        if hasattr(qpos, '__getitem__'):
                            x = float(qpos[0])
                            theta = float(qpos[1])
                        else:
                            x = 0.0
                            theta = 0.0
                    except (TypeError, ValueError, IndexError, KeyError):
                        x = 0.0
                        theta = 0.0
                    
                    try:
                        if hasattr(qvel, '__getitem__'):
                            x_dot = float(qvel[0])
                            theta_dot = float(qvel[1])
                        else:
                            x_dot = 0.0
                            theta_dot = 0.0
                    except (TypeError, ValueError, IndexError, KeyError):
                        x_dot = 0.0
                        theta_dot = 0.0
                    
                    # Preserve original observation shape and update first 4 elements
                    obs_base = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
                    obs = obs.copy()  # Make a copy to avoid modifying the original
                    obs[:4] = obs_base[:4]  # Update first 4 elements with physics values
                    
                    # Debug: Log observation construction for first few steps
                    if not hasattr(self, '_step_count'):
                        self._step_count = 0
                    self._step_count += 1
                    if self._step_count <= 5:
                        print(f"[DIAG] Step {self._step_count}: Manually constructed obs from physics: [x={x:.4f}, x_dot={x_dot:.4f}, theta={theta:.4f}, theta_dot={theta_dot:.4f}]")
                except Exception as e:
                    # If extraction fails, use observation from environment
                    if not hasattr(self, '_step_count'):
                        self._step_count = 0
                    self._step_count += 1
                    if self._step_count <= 3:
                        print(f"[DIAG] Step {self._step_count}: Could not extract physics state: {e}")
            
            return obs, reward, terminated, truncated, info
    
    env = CustomInitWrapper(env)
    
    # ✅ Compatibility wrapper: Handle reset() and step() for old gym API  
    # Apply AFTER FlattenObservation to fix API compatibility while preserving observation flattening
    class ResetAdapter(gym.Wrapper):
        def reset(self, **kwargs):
            # Handle seed parameter
            seed = kwargs.pop('seed', None)
            if seed is not None:
                # Seed the underlying environment
                if hasattr(self.env, 'seed'):
                    self.env.seed(seed)
                # Also try to seed base environment
                base = self.env
                while hasattr(base, 'env'):
                    base = base.env
                    if hasattr(base, 'seed'):
                        base.seed(seed)
            
            # Call reset on the wrapped environment (goes through CustomInitWrapper)
            reset_result = self.env.reset(**kwargs)
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            # Manually flatten if the observation is a dict
            if hasattr(self.env, 'observation_space') and isinstance(obs, dict):
                obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
            
            # Always return tuple (obs, info) for compatibility with Stable-Baselines3
            # The shimmy compatibility layer and DummyVecEnv expect this format
            return obs, info
        
        def step(self, action):
            # Initialize step counter if needed
            if not hasattr(self, '_step_count'):
                self._step_count = 0
                self._episode_count = 0
            self._step_count += 1
            
            # Call step on the wrapped environment (goes through CustomInitWrapper)
            step_result = self.env.step(action)
            
            # Handle both old gym API (4 values) and new gymnasium API (5 values)
            if len(step_result) == 5:
                obs, reward_from_env, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward_from_env, done, info = step_result
                terminated = done
                truncated = False
            else:
                raise ValueError(f"Unexpected step() return format: {len(step_result)} values")
            
            # Manually flatten observation if needed (like FlattenObservation does)
            if hasattr(self.env, 'observation_space') and isinstance(obs, dict):
                obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
            
            # ✅ dm_control2gym should already provide the native reward from dm_control
            # Use reward_from_env as the base (it should be the native reward)
            # If it's None or 0, we'll compute custom reward as fallback
            native_reward = reward_from_env if reward_from_env is not None and reward_from_env != 0.0 else None
            
            # Try to access native reward directly from dm_control if reward_from_env seems wrong
            if native_reward is None or native_reward == 0.0:
                try:
                    base_env = self.env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    
                    # Try to access dm_control environment's last time_step
                    if hasattr(base_env, '_env') and hasattr(base_env._env, '_last_time_step'):
                        time_step = base_env._env._last_time_step
                        if hasattr(time_step, 'reward') and time_step.reward is not None:
                            native_reward = float(time_step.reward)
                    elif hasattr(base_env, 'dmcenv') and hasattr(base_env.dmcenv, '_last_time_step'):
                        time_step = base_env.dmcenv._last_time_step
                        if hasattr(time_step, 'reward') and time_step.reward is not None:
                            native_reward = float(time_step.reward)
                except Exception:
                    pass  # If we can't access native reward, use custom reward
            
            # ✅ Apply custom reward shaping: Physics-consistent reward for better learning
            # Extract state variables from flattened observation
            if len(obs) >= 4:
                x = obs[0]           # cart position
                x_dot = obs[1]       # cart velocity
                theta = obs[2]       # pole angle
                theta_dot = obs[3]   # pole angular velocity
                
                # Debug: Log observation and action info (first few steps of first episode)
                if self._episode_count == 0 and self._step_count <= 10:
                    print(f"[DIAG] Step {self._step_count}: obs = [x={x:.3f}, x_dot={x_dot:.3f}, theta={theta:.3f}, theta_dot={theta_dot:.3f}]")
                    print(f"       Action received: {action} (type: {type(action)})")
                    if self._step_count == 1:
                        print(f"       Action space: {self.env.action_space if hasattr(self.env, 'action_space') else 'N/A'}")
                
                # Use native reward if available, otherwise compute custom reward
                if native_reward is not None:
                    # Use native dm_control reward as base
                    reward = native_reward
                    
                    # Add velocity penalties to discourage swinging (on top of native reward)
                    theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                    angular_velocity_penalty = 0.005 * theta_dot**2  # Small penalty for swinging
                    cart_velocity_penalty = 0.005 * x_dot**2  # Small penalty for excessive movement
                    
                    # Apply penalties (reduce reward slightly for high velocities)
                    reward = reward - angular_velocity_penalty - cart_velocity_penalty
                    reward = float(np.clip(reward, 0.0, 1.0))
                else:
                    # Fallback: custom reward function (similar to native dm_control)
                    # Normalize theta to [-π, π] for consistent angle comparison
                    theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                    
                    # Native dm_control reward formula: reward = 1.0 - (cost_angle + cost_position)
                    # Cost components (0 = perfect, larger = worse)
                    angle_cost = theta_normalized**2
                    position_cost = 0.1 * x**2
                    
                    # Velocity costs: penalize excessive swinging/movement
                    angular_velocity_cost = 0.01 * theta_dot**2
                    cart_velocity_cost = 0.01 * x_dot**2
                    
                    # Base reward: 1.0 when perfect, decreases with costs
                    base_reward = 1.0 - (angle_cost + position_cost + angular_velocity_cost + cart_velocity_cost)
                    reward = float(np.clip(base_reward, 0.0, 1.0))
                
                # Add small bonus for maintaining upright position to encourage stability
                theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                if abs(theta_normalized) < 0.15 and abs(x) < 0.3:  # Close to upright and centered
                    if not hasattr(self, "_stable_steps"):
                        self._stable_steps = 0
                    self._stable_steps += 1
                    # Small bonus that grows with stability time
                    stability_bonus = min(0.1, self._stable_steps / 500.0)
                    reward += stability_bonus
                else:
                    if hasattr(self, "_stable_steps"):
                        self._stable_steps = 0
                
                # Final reward in [0, 1.1] range
                reward = float(np.clip(reward, 0.0, 1.1))
                
                # Debug: Log reward computation (first few steps of first episode)
                if self._episode_count == 0 and self._step_count <= 10:
                    theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                    stability_bonus_val = getattr(self, "_stable_steps", 0) / 500.0 if abs(theta_normalized) < 0.15 and abs(x) < 0.3 else 0.0
                    if native_reward is not None:
                        print(f"       Reward = {reward:.4f} (native={native_reward:.4f}, bonus={stability_bonus_val:.4f})")
                    else:
                        print(f"       Reward = {reward:.4f} (custom, bonus={stability_bonus_val:.4f})")
                    print(f"       State: theta_norm={theta_normalized:.3f}, x={x:.3f}, theta_dot={theta_dot:.3f}, x_dot={x_dot:.3f}")
                
                # Terminate if cart runs off or pole falls too far
                # Note: Pole starts inverted (theta ≈ π ± 0.2), so we need to allow that initial state
                # Only terminate if cart position is out of bounds OR pole angle is extreme
                # Normalize theta to [-π, π] range for proper angle comparison
                theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                
                if abs(x) > 2.4:
                    terminated = True
                    reward -= 1.0  # small penalty for failure
                    if self._episode_count < 3:
                        print(f"[DIAG] Episode {self._episode_count + 1} terminated at step {self._step_count}: |x|={abs(x):.3f} > 2.4")
                elif abs(theta_normalized) > np.pi * 1.1:  # Terminate if pole goes beyond ~198 degrees from upright
                    # This allows starting at π (inverted, ~180°) but terminates if it goes beyond ~198°
                    # Starting range is π ± 0.2 ≈ 2.94-3.34, normalized gives ~2.94-3.14 or ~-3.14 to -2.94
                    # Threshold of π*1.1 ≈ 3.46 allows the starting state but catches extreme deviations
                    terminated = True
                    reward -= 1.0  # small penalty for failure
                    if self._episode_count < 3:
                        print(f"[DIAG] Episode {self._episode_count + 1} terminated at step {self._step_count}: |theta_normalized|={abs(theta_normalized):.3f} > {np.pi * 1.1:.3f}")
            
            # Track episode completion
            if terminated or truncated:
                # Reset hold_steps counter at episode end
                if hasattr(self, "_hold_steps"):
                    self._hold_steps = 0
                self._episode_count += 1
                if self._episode_count <= 3:
                    print(f"[DIAG] Episode {self._episode_count} ended: terminated={terminated}, truncated={truncated}, total_steps={self._step_count}")
                self._step_count = 0
            
            return obs, reward, terminated, truncated, info
    
    env = ResetAdapter(env)
    
    # ✅ Convert gym spaces to gymnasium spaces for Stable-Baselines3 2.0.0 compatibility
    class SpaceConverter(gym.Wrapper):
        """Convert gym spaces to gymnasium spaces"""
        def __init__(self, env):
            super().__init__(env)
            # Convert action space
            if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                # Box space
                self.action_space = gymnasium.spaces.Box(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    shape=env.action_space.shape,
                    dtype=env.action_space.dtype
                )
            elif hasattr(env.action_space, 'n'):
                # Discrete space
                self.action_space = gymnasium.spaces.Discrete(env.action_space.n)
            else:
                self.action_space = env.action_space
            
            # Convert observation space
            if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
                # Box space
                self.observation_space = gymnasium.spaces.Box(
                    low=env.observation_space.low,
                    high=env.observation_space.high,
                    shape=env.observation_space.shape,
                    dtype=env.observation_space.dtype
                )
            elif hasattr(env.observation_space, 'n'):
                # Discrete space
                self.observation_space = gymnasium.spaces.Discrete(env.observation_space.n)
            else:
                self.observation_space = env.observation_space
    
    env = SpaceConverter(env)

    if hasattr(env, "seed"):
        env.seed(seed)
    np.random.seed(seed)
    return env



def parse_monitor_logs(log_file):
    """Parse Monitor log file to extract episode rewards and lengths"""
    import pandas as pd
    try:
        df = pd.read_csv(log_file, skiprows=1)
        episode_rewards = df['r'].values.tolist()
        episode_lengths = df['l'].values.tolist()
        timesteps = df['t'].values.tolist()
        return episode_rewards, episode_lengths, timesteps
    except Exception as e:
        print(f"Warning: Could not parse monitor log: {e}")
        return [], [], []


def train(seed=0, total_timesteps=1_000_000, save_dir='results/models', log_dir='results/training_logs'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    set_seed(seed)

    # Define monitor file path in outer scope so it can be accessed later
    monitor_file = os.path.join(log_dir, f"monitor_seed{seed}.csv")

    # Create environment with Monitor wrapper to log episode rewards
    def _make_monitored_env():
        env = make_env(seed)
        env = Monitor(env, monitor_file, allow_early_resets=True)
        return env
    
    env = DummyVecEnv([_make_monitored_env])
    
    # ✅ Diagnostic: Check environment setup
    print("\n[DIAG] Environment Diagnostics:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    sample_action = env.action_space.sample()
    print(f"  Sample action: {sample_action} (type: {type(sample_action)}, shape: {sample_action.shape if hasattr(sample_action, 'shape') else 'N/A'})")
    
    # Test reset and first step (DummyVecEnv returns arrays)
    test_obs = env.reset()
    if hasattr(test_obs, '__getitem__') and len(test_obs) > 0:
        obs_flat = test_obs[0] if hasattr(test_obs[0], '__len__') else test_obs
        print(f"  Initial observation shape: {obs_flat.shape if hasattr(obs_flat, 'shape') else len(obs_flat)}")
        if hasattr(obs_flat, '__getitem__') and len(obs_flat) >= 4:
            print(f"  Initial observation [x, x_dot, theta, theta_dot]: [{obs_flat[0]:.4f}, {obs_flat[1]:.4f}, {obs_flat[2]:.4f}, {obs_flat[3]:.4f}]")
            print(f"  Expected: theta should be ≈ π ({np.pi:.3f}) if physics modification worked")
    
    test_obs, test_reward, test_done, test_info = env.step([sample_action])
    if hasattr(test_reward, '__getitem__'):
        print(f"  First step reward: {test_reward[0]:.4f}")
        print(f"  First step terminated: {test_done[0] if hasattr(test_done, '__getitem__') else test_done}")
    if hasattr(test_obs, '__getitem__') and len(test_obs) > 0:
        obs_flat = test_obs[0] if hasattr(test_obs[0], '__len__') else test_obs
        if hasattr(obs_flat, '__getitem__') and len(obs_flat) >= 4:
            print(f"  First step observation [x, x_dot, theta, theta_dot]: [{obs_flat[0]:.4f}, {obs_flat[1]:.4f}, {obs_flat[2]:.4f}, {obs_flat[3]:.4f}]")
    print("[DIAG] End diagnostics\n")
    
    # Network architecture matching reference code (ppo.py)
    # Actor: [64, 64] with Tanh, Critic: [64, 64] with Tanh
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=[64, 64]  # Match reference code architecture
    )

    # Hyperparameters matching reference code (ppo.py)
    # Learning rate: 3e-4 (fixed, not scheduled)
    # Clip ratio: 0.2 (fixed, not scheduled)
    # Batch size: 64 (match reference)
    # Entropy coefficient: 0.001 (match reference)
    # Note: Reference code does per-episode updates (collect episode, then update)
    # Stable-Baselines3 uses n_steps to collect steps before updating
    # Using smaller n_steps (512-1024) to update more frequently, closer to per-episode behavior
    # Episodes are typically 100-1000 steps, so 512-1024 is a reasonable approximation
    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,  # Match reference: 3e-4 (fixed)
                gamma=0.99,  # Match reference
                gae_lambda=0.95,  # Match reference (lambda in reference)
                clip_range=0.2,  # Match reference: 0.2 (fixed, not scheduled)
                n_steps=512,  # Smaller buffer to update more frequently (closer to per-episode updates)
                batch_size=64,  # Match reference: 64
                n_epochs=10,  # Match reference
                ent_coef=0.001,  # Match reference: 0.001
                vf_coef=0.5,  # Standard value (not specified in reference)
                max_grad_norm=0.5,  # Match reference
                normalize_advantage=True,  # Helpful for stability
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=seed)

    # Add callback to monitor training progress
    class RewardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.last_printed_reward = None
            
        def _on_step(self) -> bool:
            # Check if there are episode infos
            if 'episode' in self.locals.get('infos', [{}])[0]:
                ep_info = self.locals['infos'][0]['episode']
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                    # Only print every 200 episodes, or when reward changes significantly
                    if len(self.episode_rewards) % 200 == 0:
                        mean_reward = np.mean(self.episode_rewards[-10:])
                        # Only print if reward changed significantly (more than 0.1 difference)
                        if self.last_printed_reward is None or abs(mean_reward - self.last_printed_reward) > 0.1:
                            print(f"  Recent 10 episodes mean reward: {mean_reward:.2f}")
                            self.last_printed_reward = mean_reward
            return True
    
    callback = RewardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(save_dir, f"ppo_cartpole_seed{seed}.zip"))
    
    # Parse and save training logs in a more usable format
    # Monitor wrapper appends .monitor.csv to the filename
    actual_monitor_file = f"{monitor_file}.monitor.csv"
    episode_rewards, episode_lengths, timesteps = parse_monitor_logs(actual_monitor_file)
    
    # ✅ Diagnostic: Check episode termination
    if episode_lengths:
        max_ep_len = max(episode_lengths) if episode_lengths else 0
        min_ep_len = min(episode_lengths) if episode_lengths else 0
        mean_ep_len = np.mean(episode_lengths) if episode_lengths else 0
        episodes_terminated_early = sum(1 for l in episode_lengths if l < max_ep_len)
        print(f"\n[DIAG] Episode Length Analysis:")
        print(f"  Total episodes: {len(episode_lengths)}")
        print(f"  Max episode length: {max_ep_len}")
        print(f"  Min episode length: {min_ep_len}")
        print(f"  Mean episode length: {mean_ep_len:.1f}")
        print(f"  Episodes terminated early (< {max_ep_len}): {episodes_terminated_early} ({100*episodes_terminated_early/len(episode_lengths):.1f}%)")
        if episodes_terminated_early == 0:
            print(f"  ⚠️  WARNING: No episodes terminated early - environment may not be terminating properly!")
        if episode_rewards:
            print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"  Reward range: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
    
    log_file = os.path.join(log_dir, f"training_seed{seed}.json")
    data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'timesteps': timesteps
    }
    with open(log_file, 'w') as f:
        json.dump(data, f)
    print(f"✅ Training logs saved to {log_file}")
    
    env.close()
    
    return data


if __name__ == "__main__":
    # Train on seeds 0, 1, 2 as required
    training_seeds = [0, 1, 2]
    for seed in training_seeds:
        print(f"\n{'='*50}")
        print(f"Training with seed {seed}")
        print(f"{'='*50}")
        train(seed=seed)
    print(f"\n✅ Training completed for all seeds: {training_seeds}")
