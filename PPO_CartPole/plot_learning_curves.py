import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import moving_average

def load_training_logs(log_dir='results/training_logs', seeds=[0, 1, 2]):
    """Load training logs for multiple seeds"""
    all_data = {}
    for seed in seeds:
        log_file = os.path.join(log_dir, f"training_seed{seed}.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                all_data[seed] = json.load(f)
        else:
            print(f"⚠️  Training log not found: {log_file}")
    return all_data

def load_evaluation_logs(log_dir='results/evaluation_logs', eval_seed=10, training_seeds=[0, 1, 2]):
    """Load evaluation logs for multiple models"""
    all_data = {}
    for seed in training_seeds:
        log_file = os.path.join(log_dir, f"evaluation_seed{eval_seed}.json")
        # Try to load from combined file first
        combined_file = os.path.join(log_dir, f"evaluation_all_seed{eval_seed}.json")
        if os.path.exists(combined_file):
            with open(combined_file, 'r') as f:
                combined = json.load(f)
                if f'seed{seed}' in combined:
                    all_data[seed] = combined[f'seed{seed}']
        elif os.path.exists(log_file):
            with open(log_file, 'r') as f:
                all_data[seed] = json.load(f)
        else:
            print(f"⚠️  Evaluation log not found for seed {seed}")
    return all_data

def plot_learning_curves(training_data, eval_data=None, save_path='results/learning_curves.png'):
    """Plot learning curves with mean ± std for training and evaluation"""
    fig, axes = plt.subplots(1, 2 if eval_data else 1, figsize=(14, 5) if eval_data else (7, 5))
    if not eval_data:
        axes = [axes]
    
    # Plot training curves
    seeds = sorted(training_data.keys())
    if len(seeds) == 0:
        print("⚠️  No training data to plot")
        return
    
    # Extract episode rewards for each seed
    all_rewards = []
    max_episodes = 0
    for seed in seeds:
        rewards = training_data[seed].get('episode_rewards', [])
        all_rewards.append(rewards)
        max_episodes = max(max_episodes, len(rewards))
    
    # Pad to same length
    for i in range(len(all_rewards)):
        if len(all_rewards[i]) < max_episodes:
            all_rewards[i] = all_rewards[i] + [all_rewards[i][-1]] * (max_episodes - len(all_rewards[i]))
    
    # Convert to numpy array and compute mean/std
    rewards_array = np.array(all_rewards)
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    
    # Convert episode numbers to update numbers (assuming ~100 episodes per update on average)
    # Or use timesteps if available
    episodes = np.arange(1, len(mean_rewards) + 1)
    
    # Use timesteps if available from first seed
    if 'timesteps' in training_data[seeds[0]] and len(training_data[seeds[0]]['timesteps']) > 0:
        # Interpolate to get updates
        timesteps = np.array(training_data[seeds[0]]['timesteps'])
        # Convert to updates (assuming n_steps=2048)
        updates = timesteps / 2048
        x_axis = updates
        x_label = 'Learning Updates'
    else:
        # Estimate updates from episodes (rough approximation)
        x_axis = episodes / 100  # Rough estimate: ~100 episodes per update
        x_label = 'Learning Updates (estimated)'
    
    # Plot training curve
    axes[0].plot(x_axis, mean_rewards, label='Mean Training Reward', color='blue', linewidth=2)
    axes[0].fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                        alpha=0.3, color='blue', label='±1 Std Dev')
    axes[0].set_xlabel(x_label, fontsize=12)
    axes[0].set_ylabel('Episode Reward', fontsize=12)
    axes[0].set_title('Training Learning Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot evaluation curve if available
    if eval_data and len(eval_data) > 0:
        eval_seeds = sorted(eval_data.keys())
        eval_rewards = [eval_data[seed]['mean_reward'] for seed in eval_seeds]
        eval_stds = [eval_data[seed]['std_reward'] for seed in eval_seeds]
        
        # For evaluation, we show mean across all training seeds
        eval_mean = np.mean(eval_rewards)
        eval_std = np.std(eval_rewards)  # Std across different training seeds
        
        # Evaluation is typically done at the end, so we show it as a point or bar
        # Or we can show it as a horizontal line
        axes[1].axhline(y=eval_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean Eval Reward: {eval_mean:.2f}')
        axes[1].fill_between([0, x_axis[-1]], eval_mean - eval_std, eval_mean + eval_std,
                           alpha=0.3, color='red', label='±1 Std Dev')
        
        # Also plot individual evaluation points
        for i, seed in enumerate(eval_seeds):
            axes[1].scatter(x_axis[-1], eval_rewards[i], s=50, alpha=0.6, 
                          label=f'Seed {seed}: {eval_rewards[i]:.2f}')
        
        axes[1].set_xlabel(x_label, fontsize=12)
        axes[1].set_ylabel('Episode Reward', fontsize=12)
        axes[1].set_title('Evaluation Results (Seed 10)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, x_axis[-1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Learning curves saved to {save_path}")
    plt.close()

def plot_combined_curves(training_data, eval_data=None, save_path='results/learning_curves_combined.png'):
    """Plot combined training and evaluation curves on same plot"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    seeds = sorted(training_data.keys())
    if len(seeds) == 0:
        print("⚠️  No training data to plot")
        return
    
    # Extract episode rewards for each seed
    all_rewards = []
    max_episodes = 0
    for seed in seeds:
        rewards = training_data[seed].get('episode_rewards', [])
        all_rewards.append(rewards)
        max_episodes = max(max_episodes, len(rewards))
    
    # Pad to same length
    for i in range(len(all_rewards)):
        if len(all_rewards[i]) < max_episodes:
            all_rewards[i] = all_rewards[i] + [all_rewards[i][-1]] * (max_episodes - len(all_rewards[i]))
    
    # Convert to numpy array and compute mean/std
    rewards_array = np.array(all_rewards)
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    
    # Get x-axis (updates)
    episodes = np.arange(1, len(mean_rewards) + 1)
    if 'timesteps' in training_data[seeds[0]] and len(training_data[seeds[0]]['timesteps']) > 0:
        timesteps = np.array(training_data[seeds[0]]['timesteps'])
        updates = timesteps / 2048
        x_axis = updates
        x_label = 'Learning Updates'
    else:
        x_axis = episodes / 100
        x_label = 'Learning Updates (estimated)'
    
    # Plot training curve
    ax.plot(x_axis, mean_rewards, label='Training (mean ± std)', color='blue', linewidth=2)
    ax.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                   alpha=0.3, color='blue')
    
    # Plot evaluation if available
    if eval_data and len(eval_data) > 0:
        eval_seeds = sorted(eval_data.keys())
        eval_rewards = [eval_data[seed]['mean_reward'] for seed in eval_seeds]
        eval_stds = [eval_data[seed]['std_reward'] for seed in eval_seeds]
        
        eval_mean = np.mean(eval_rewards)
        eval_std = np.std(eval_rewards)
        
        # Show evaluation as a horizontal band at the end
        ax.axhline(y=eval_mean, color='red', linestyle='--', linewidth=2, 
                  label=f'Evaluation (mean ± std): {eval_mean:.2f} ± {eval_std:.2f}')
        ax.fill_between([x_axis[-1]*0.9, x_axis[-1]], eval_mean - eval_std, eval_mean + eval_std,
                       alpha=0.3, color='red')
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Learning Curves: Training and Evaluation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Combined learning curves saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Load training and evaluation data
    training_data = load_training_logs(seeds=[0, 1, 2])
    eval_data = load_evaluation_logs(eval_seed=10, training_seeds=[0, 1, 2])
    
    if len(training_data) > 0:
        # Plot separate curves
        plot_learning_curves(training_data, eval_data, save_path='results/learning_curves.png')
        # Plot combined curve
        plot_combined_curves(training_data, eval_data, save_path='results/learning_curves_combined.png')
        print("\n✅ All plots generated successfully!")
    else:
        print("⚠️  No training data found. Please run train.py first.")

