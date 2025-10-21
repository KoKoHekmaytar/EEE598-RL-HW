"""
DQN_gridworld_compare.py
Run DQN on the 3x4 stochastic gridworld with different hyperparameters (alpha, gamma, epsilon schedules)
and compare the learning curves.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime
from DQN_gridworld import GridWorld3x4, DQNHyper, train_dqn, evaluate_policy

# ------------------------------------------
# Parameter sweep setup
# ------------------------------------------
def run_experiments():
    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"dqn_comparison_results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving comparison results to folder: {output_folder}")
    
    env = GridWorld3x4(penalty_minus_one=True, step_cost=-0.04, seed=123)

    configs = [
        # (label, hyperparams)
        ("α=1e-3, γ=0.99", DQNHyper(lr=1e-3, gamma=0.99, eval_frequency=200, eval_episodes=50)),
        ("α=5e-4, γ=0.99", DQNHyper(lr=5e-4, gamma=0.99, eval_frequency=200, eval_episodes=50)),
        ("α=1e-3, γ=0.95", DQNHyper(lr=1e-3, gamma=0.95, eval_frequency=200, eval_episodes=50)),
        ("α=2e-3, γ=0.99", DQNHyper(lr=2e-3, gamma=0.99, eval_frequency=200, eval_episodes=50)),
        ("α=1e-3, γ=0.90", DQNHyper(lr=1e-3, gamma=0.90, eval_frequency=200, eval_episodes=50)),
    ]

    results = []
    all_stats = []
    
    for i, (label, hp) in enumerate(configs):
        print(f"\n=== Running {label} ({i+1}/{len(configs)}) ===")
        stats = train_dqn(env, hp, verbose=True)
        
        # Store comprehensive results
        result_data = {
            'label': label,
            'hyperparams': {
                'lr': hp.lr,
                'gamma': hp.gamma,
                'eps_start': hp.eps_start,
                'eps_end': hp.eps_end,
                'eps_decay_episodes': hp.eps_decay_episodes,
                'batch_size': hp.batch_size,
                'buffer_size': hp.buffer_size,
                'hidden_size': hp.hidden_size,
                'target_update_every': hp.target_update_every
            },
            'training_returns': stats['returns'].tolist(),
            'training_steps': stats['steps_hist'].tolist(),
            'eps_history': stats['eps_hist'].tolist(),
            'loss_history': stats['loss_hist'].tolist(),
            'eval_returns': stats['eval_returns'].tolist(),
            'eval_steps': stats['eval_steps'].tolist(),
            'eval_episodes': stats['eval_episodes'].tolist(),
            'final_training_return': float(np.mean(stats['returns'][-100:])) if len(stats['returns']) >= 100 else float(np.mean(stats['returns'])),
            'final_eval_return': float(stats['eval_returns'][-1]) if len(stats['eval_returns']) > 0 else None,
            'total_episodes': len(stats['returns']),
            'total_gradient_steps': len(stats['loss_hist'])
        }
        
        results.append((label, stats['returns'], stats['eps_hist'], stats['eval_returns'], stats['eval_episodes']))
        all_stats.append(result_data)
        
        # Save individual experiment results
        with open(f"{output_folder}/experiment_{i+1}_{label.replace(',', '_').replace('=', '_')}.json", 'w') as f:
            json.dump(result_data, f, indent=2)
    
    # Save all results summary
    with open(f"{output_folder}/all_experiments_summary.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    return results, output_folder, all_stats

# ------------------------------------------
# Plotting comparisons
# ------------------------------------------
def plot_comprehensive_comparisons(results, output_folder):
    """Create comprehensive comparison plots with training and validation curves"""
    
    # 1. Training vs Validation Returns Comparison
    plt.figure(figsize=(15, 10))
    
    # Training returns with moving average
    plt.subplot(2, 3, 1)
    for label, returns, _, eval_returns, eval_episodes in results:
        if len(returns) >= 50:
            ma = np.convolve(returns, np.ones(50)/50, mode='valid')
            plt.plot(range(49, len(returns)), ma, label=f'{label} (training)', alpha=0.7)
        else:
            plt.plot(returns, label=f'{label} (training)', alpha=0.7)
        
        # Add evaluation points
        if len(eval_returns) > 0:
            plt.plot(eval_episodes, eval_returns, 'o-', label=f'{label} (validation)', 
                    markersize=4, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training vs Validation Returns")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Final Performance Comparison
    plt.subplot(2, 3, 2)
    labels = []
    final_training = []
    final_validation = []
    
    for label, returns, _, eval_returns, _ in results:
        labels.append(label)
        final_training.append(np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns))
        final_validation.append(eval_returns[-1] if len(eval_returns) > 0 else 0)
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, final_training, width, label='Final Training Return', alpha=0.8)
    plt.bar(x + width/2, final_validation, width, label='Final Validation Return', alpha=0.8)
    
    plt.xlabel("Configuration")
    plt.ylabel("Return")
    plt.title("Final Performance Comparison")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Learning Curves (Raw)
    plt.subplot(2, 3, 3)
    for label, returns, _, _, _ in results:
        plt.plot(returns, label=label, alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Raw Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Validation Curves Only
    plt.subplot(2, 3, 4)
    for label, _, _, eval_returns, eval_episodes in results:
        if len(eval_returns) > 0:
            plt.plot(eval_episodes, eval_returns, 'o-', label=label, markersize=6, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Validation Return")
    plt.title("Validation Performance Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Epsilon Schedules
    plt.subplot(2, 3, 5)
    for label, _, eps, _, _ in results:
        plt.plot(eps, label=label)
    plt.xlabel("Episode")
    plt.ylabel("ε (exploration rate)")
    plt.title("Epsilon Schedule Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary table
    table_data = []
    for i, (label, returns, _, eval_returns, _) in enumerate(results):
        final_train = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        final_eval = eval_returns[-1] if len(eval_returns) > 0 else 0
        max_train = np.max(returns)
        max_eval = np.max(eval_returns) if len(eval_returns) > 0 else 0
        
        table_data.append([
            label,
            f"{final_train:.3f}",
            f"{final_eval:.3f}",
            f"{max_train:.3f}",
            f"{max_eval:.3f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Config', 'Final Train', 'Final Eval', 'Max Train', 'Max Eval'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.title("Performance Summary")
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive comparison to {output_folder}/comprehensive_comparison.png")

def plot_individual_learning_curves(results, output_folder):
    """Create individual learning curve plots for each configuration"""
    
    for i, (label, returns, eps_hist, eval_returns, eval_episodes) in enumerate(results):
        plt.figure(figsize=(12, 8))
        
        # Training returns
        plt.subplot(2, 2, 1)
        plt.plot(returns, label='Training Returns', alpha=0.7)
        if len(returns) >= 50:
            ma = np.convolve(returns, np.ones(50)/50, mode='valid')
            plt.plot(range(49, len(returns)), ma, label='MA(50)', linewidth=2)
        
        if len(eval_returns) > 0:
            plt.plot(eval_episodes, eval_returns, 'o-', label='Validation Returns', 
                    color='red', linewidth=2, markersize=4)
        
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title(f'{label} - Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Epsilon schedule
        plt.subplot(2, 2, 2)
        plt.plot(eps_hist)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title(f'{label} - Epsilon Schedule')
        plt.grid(True, alpha=0.3)
        
        # Performance metrics
        plt.subplot(2, 2, 3)
        final_train = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        final_eval = eval_returns[-1] if len(eval_returns) > 0 else 0
        max_train = np.max(returns)
        max_eval = np.max(eval_returns) if len(eval_returns) > 0 else 0
        
        metrics = ['Final Train', 'Final Eval', 'Max Train', 'Max Eval']
        values = [final_train, final_eval, max_train, max_eval]
        colors = ['blue', 'red', 'green', 'orange']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.ylabel('Return')
        plt.title(f'{label} - Performance Metrics')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        
        # Training vs Validation comparison
        plt.subplot(2, 2, 4)
        if len(eval_returns) > 0:
            plt.plot(eval_episodes, eval_returns, 'o-', label='Validation', 
                    color='red', linewidth=2, markersize=6)
        plt.plot(returns, label='Training', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title(f'{label} - Training vs Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_label = label.replace(',', '_').replace('=', '_').replace('α', 'alpha').replace('γ', 'gamma')
        plt.savefig(f'{output_folder}/individual_curve_{i+1}_{safe_label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved individual curve for {label} to {output_folder}/individual_curve_{i+1}_{safe_label}.png")

def create_summary_csv(all_stats, output_folder):
    """Create a summary CSV with all experiment results"""
    summary_data = []
    
    for stats in all_stats:
        summary_data.append({
            'Configuration': stats['label'],
            'Learning_Rate': stats['hyperparams']['lr'],
            'Gamma': stats['hyperparams']['gamma'],
            'Final_Training_Return': stats['final_training_return'],
            'Final_Evaluation_Return': stats['final_eval_return'],
            'Max_Training_Return': max(stats['training_returns']),
            'Max_Evaluation_Return': max(stats['eval_returns']) if len(stats['eval_returns']) > 0 else None,
            'Total_Episodes': stats['total_episodes'],
            'Total_Gradient_Steps': stats['total_gradient_steps'],
            'Epsilon_Start': stats['hyperparams']['eps_start'],
            'Epsilon_End': stats['hyperparams']['eps_end'],
            'Batch_Size': stats['hyperparams']['batch_size'],
            'Buffer_Size': stats['hyperparams']['buffer_size'],
            'Hidden_Size': stats['hyperparams']['hidden_size']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{output_folder}/experiment_summary.csv', index=False)
    print(f"Saved experiment summary to {output_folder}/experiment_summary.csv")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

# ------------------------------------------
# Main entry
# ------------------------------------------
if __name__ == "__main__":
    results, output_folder, all_stats = run_experiments()
    
    # Create comprehensive comparison plots
    plot_comprehensive_comparisons(results, output_folder)
    
    # Create individual learning curve plots
    plot_individual_learning_curves(results, output_folder)
    
    # Create summary CSV
    create_summary_csv(all_stats, output_folder)
    
    print(f"\nAll comparison results saved to folder: {output_folder}")
    print("Files created:")
    print(f"  - {output_folder}/comprehensive_comparison.png")
    print(f"  - {output_folder}/individual_curve_*.png (one for each configuration)")
    print(f"  - {output_folder}/experiment_summary.csv")
    print(f"  - {output_folder}/all_experiments_summary.json")
    print(f"  - {output_folder}/experiment_*.json (individual experiment results)")
