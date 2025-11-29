import os
import numpy as np
import json

# 🩹 Compatibility patch for deprecated np.int - MUST be before any other imports
if not hasattr(np, "int"):
    np.int = int

import gym  # keep gym, not gymnasium, because dm_control2gym uses gym API
import gymnasium
from stable_baselines3 import PPO
from utils import set_seed
import imageio

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
                    
                    # Debug: Verify physics modification worked (disabled for cleaner output)
                    # if hasattr(physics, "data") and hasattr(physics.data, "qpos"):
                    #     qpos_after = physics.data.qpos
                    #     if hasattr(qpos_after, '__len__') and len(qpos_after) > 1:
                    #         # Print first few resets to verify
                    #         if not hasattr(self, '_reset_count'):
                    #             self._reset_count = 0
                    #         self._reset_count += 1
                    #         if self._reset_count <= 3:
                    #             print(f"Reset {self._reset_count}: qpos after modification = {qpos_after[:2] if hasattr(qpos_after, '__getitem__') else qpos_after}")
                    
                    # Get updated observation after modifying physics state
                    # ALWAYS manually construct from physics since task.get_observation() returns zeros
                    # Extract state directly from physics: [x, x_dot, theta, theta_dot]
                    qpos = physics.data.qpos
                    qvel = physics.data.qvel
                    
                    # Extract values - handle both array and dict-like structures
                    try:
                        if hasattr(qpos, '__getitem__'):
                            x = float(qpos[0])
                            theta = float(qpos[1])
                        else:
                            x = 0.0
                            theta = float(pole_angle)
                    except (TypeError, ValueError, IndexError, KeyError):
                        x = 0.0
                        theta = float(pole_angle)
                    
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
                    obs_base = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
                    obs = obs.copy()  # Make a copy to avoid modifying the original
                    obs[:4] = obs_base[:4]  # Update first 4 elements with physics values
                except Exception as e:
                    # If modification fails, use original observation
                    # Suppress warning for OrderedDict issues - physics modification is optional
                    if "OrderedDict" not in str(e):
                        print(f"Warning: Could not modify initial state: {e}")
            
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
                except Exception:
                    # If extraction fails, use observation from environment
                    pass
            
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
            
            # Always return tuple (obs, info) for compatibility
            return obs, info
        
        def step(self, action):
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
            
            # ✅ Apply custom reward shaping: Explicitly encourage reaching and holding upright position
            # Extract state variables from flattened observation
            if len(obs) >= 4:
                x = obs[0]           # cart position
                x_dot = obs[1]       # cart velocity
                theta = obs[2]       # pole angle
                theta_dot = obs[3]   # pole angular velocity
                
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
                
                # Terminate if cart runs off or pole falls too far
                # Note: Pole starts inverted (theta ≈ π ± 0.2), so we need to allow that initial state
                # Only terminate if cart position is out of bounds OR pole angle is extreme
                # Normalize theta to [-π, π] range for proper angle comparison
                theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
                
                if abs(x) > 2.4:
                    terminated = True
                    reward -= 1.0  # small penalty for failure
                elif abs(theta_normalized) > np.pi * 1.1:  # Terminate if pole goes beyond ~198 degrees from upright
                    # This allows starting at π (inverted, ~180°) but terminates if it goes beyond ~198°
                    # Starting range is π ± 0.2 ≈ 2.94-3.34, normalized gives ~2.94-3.14 or ~-3.14 to -2.94
                    # Threshold of π*1.1 ≈ 3.46 allows the starting state but catches extreme deviations
                    terminated = True
                    reward -= 1.0  # small penalty for failure
                
                # Reset hold_steps counter at episode end
                if terminated or truncated:
                    if hasattr(self, "_hold_steps"):
                        self._hold_steps = 0
            
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

def evaluate(model_path, eval_seed=10, n_episodes=10, log_dir='results/evaluation_logs'):
    """Evaluate a trained model and log metrics"""
    os.makedirs(log_dir, exist_ok=True)
    set_seed(eval_seed)
    
    env = make_env(seed=eval_seed)
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}/{n_episodes}: Reward = {total_reward:.2f}, Length = {steps}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"\nEvaluation Results (seed {eval_seed}):")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mean Length: {mean_length:.2f} ± {std_length:.2f}")
    
    # Save evaluation logs
    log_file = os.path.join(log_dir, f"evaluation_seed{eval_seed}.json")
    data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(mean_length),
        'std_length': float(std_length),
        'n_episodes': n_episodes
    }
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Evaluation logs saved to {log_file}")
    
    env.close()
    return data


def record_video(model_path, video_path="results/videos/cartpole.mp4", n_episodes=3, seed=10):
    """Record a video of the trained model"""
    import warnings
    import sys
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress stderr output for MjrContext cleanup errors (harmless when rendering unavailable)
    # These occur during garbage collection and can't be caught normally
    original_stderr = sys.stderr
    class FilteredStderr:
        def __init__(self, original):
            self.original = original
        def write(self, text):
            # Filter out MjrContext cleanup errors
            if "MjrContext" in text and ("_ptr" in text or "__del__" in text):
                return  # Suppress these messages
            self.original.write(text)
        def flush(self):
            self.original.flush()
        def __getattr__(self, name):
            return getattr(self.original, name)
    
    sys.stderr = FilteredStderr(original_stderr)
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Note: Video recording requires OpenGL/EGL/OSMesa rendering backend
    # We'll test rendering capability when we create the environment
    
    env = None
    try:
        env = make_env(seed=seed)
        model = PPO.load(model_path)
        frames = []

        # Navigate through wrappers to get the base dm_control environment
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Get the physics object from dm_control2gym wrapper
        physics = None
        try:
            if hasattr(base_env, '_env') and hasattr(base_env._env, 'physics'):
                physics = base_env._env.physics
            elif hasattr(base_env, 'physics'):
                physics = base_env.physics
            elif hasattr(base_env, 'dmcenv') and hasattr(base_env.dmcenv, 'physics'):
                physics = base_env.dmcenv.physics
        except Exception:
            pass
        
        # Test rendering capability before proceeding
        use_physics_render = False
        if physics is not None:
            try:
                # Try a test render with error handling
                # This will fail if OpenGL/EGL/OSMesa is not available
                test_frame = physics.render(camera_id=0, width=640, height=480)
                if test_frame is not None and len(test_frame.shape) == 3:
                    use_physics_render = True
                    print("✅ Rendering available - proceeding with video recording")
            except Exception as e:
                # OpenGL/rendering not available - skip video recording gracefully
                error_msg = str(e)
                if "OpenGL" in error_msg or "mjr_makeContext" in error_msg:
                    print("⚠️  Video recording requires OpenGL/EGL/OSMesa rendering backend")
                    print("   Rendering is not available in this environment")
                    print("   To enable video recording:")
                    print("   1. Install OSMesa: sudo apt-get install libosmesa6-dev")
                    print("   2. Or use virtual display: xvfb-run -a python evaluate.py")
                else:
                    print(f"⚠️  Rendering test failed: {error_msg}")
                print("   Skipping video recording")
                # Close environment immediately to prevent cleanup issues
                try:
                    env.close()
                except Exception:
                    pass
                env = None  # Mark as closed
                return
        else:
            print("⚠️  Could not access physics object for rendering")
            print("   Skipping video recording")
            # Close environment immediately
            try:
                env.close()
            except Exception:
                pass
            env = None  # Mark as closed
            return

        # Only proceed if we can render
        if not use_physics_render:
            print("⚠️  Could not access physics for rendering")
            print("   Skipping video recording")
            return
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            step_count = 0
            while not done and step_count < 1000:  # Limit steps to prevent infinite loops
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                
                # Render frame using physics
                frame = None
                try:
                    if physics is not None:
                        frame = physics.render(camera_id=0, width=640, height=480)
                except Exception as e:
                    # If rendering fails during episode, stop recording
                    if step_count == 1 and ep == 0:
                        print(f"⚠️  Rendering failed during recording: {e}")
                        print("   Stopping video recording")
                    break
                
                if frame is not None and len(frame.shape) == 3:
                    frames.append(frame)
                
                # Limit frames to prevent memory issues
                if len(frames) >= 3000:
                    break
        
        if frames:
            try:
                imageio.mimsave(video_path, frames, fps=30)
                print(f"✅ Video saved at {video_path}")
                print(f"   Recorded {len(frames)} frames from {n_episodes} episodes")
            except ValueError as e:
                # If MP4 fails, try GIF format as fallback
                if video_path.endswith('.mp4'):
                    gif_path = video_path.replace('.mp4', '.gif')
                    try:
                        imageio.mimsave(gif_path, frames, fps=30)
                        print(f"✅ Video saved as GIF at {gif_path} (MP4 format not available)")
                        print(f"   Recorded {len(frames)} frames from {n_episodes} episodes")
                    except Exception as e2:
                        print(f"❌ Failed to save video: {e2}")
                        print(f"   Tried both MP4 and GIF formats")
                else:
                    print(f"❌ Failed to save video: {e}")
            except Exception as e:
                print(f"❌ Failed to save video: {e}")
        else:
            print("⚠️  Could not render frames - no frames collected")
            print("   This usually means OpenGL/rendering backend is not available")
    except KeyboardInterrupt:
        print("\n⚠️  Video recording interrupted by user")
    except Exception as e:
        print(f"❌ Error during video recording: {e}")
        # Don't print full traceback for known rendering issues
        if "OpenGL" not in str(e) and "mjr_makeContext" not in str(e):
            import traceback
            traceback.print_exc()
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        
        # Clean up environment safely
        # Suppress any cleanup errors from MjrContext (these are harmless when rendering isn't available)
        if env is not None:
            try:
                env.close()
            except (AttributeError, Exception):
                # Suppress all cleanup errors - MjrContext cleanup failures are harmless
                # when rendering backend is not available
                pass


if __name__ == "__main__":
    # Evaluate all trained models on seed 10
    eval_seed = 10
    training_seeds = [0, 1, 2]
    
    print(f"Evaluating models on seed {eval_seed}")
    print("="*50)
    
    all_eval_data = {}
    for seed in training_seeds:
        model_path = f"results/models/ppo_cartpole_seed{seed}.zip"
        if os.path.exists(model_path):
            print(f"\nEvaluating model trained with seed {seed}:")
            eval_data = evaluate(model_path, eval_seed=eval_seed, n_episodes=10)
            all_eval_data[f'seed{seed}'] = eval_data
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Save combined evaluation results
    if all_eval_data:
        combined_file = f"results/evaluation_logs/evaluation_all_seed{eval_seed}.json"
        with open(combined_file, 'w') as f:
            json.dump(all_eval_data, f, indent=2)
        print(f"\n✅ Combined evaluation results saved to {combined_file}")
    
    # Record video for the first available model
    for seed in training_seeds:
        model_path = f"results/models/ppo_cartpole_seed{seed}.zip"
        if os.path.exists(model_path):
            print(f"\n{'='*50}")
            print(f"Recording video for model trained with seed {seed}")
            print(f"{'='*50}")
            video_path = f"results/videos/cartpole_seed{seed}.mp4"
            record_video(model_path, video_path=video_path, n_episodes=3, seed=eval_seed)
            break
