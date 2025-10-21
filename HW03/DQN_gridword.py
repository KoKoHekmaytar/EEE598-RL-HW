"""
DQN_gridworld.py

DQN (deep Q-network) applied to the 3x4 stochastic gridworld from your assignment.
- One-hot state encoding (12 states)
- Small MLP outputs 4 Q-values
- Replay buffer, target network, minibatch training
- Epsilon schedules (linear decay)
- Plots:
    * learning curve (episode returns)
    * episode lengths
    * epsilon schedule
    * value heatmap and greedy policy
    * greedy trajectory after training

Usage:
    python DQN_gridworld.py
"""
import os
# Fix OpenMP error by allowing duplicate libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import random
from dataclasses import dataclass
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime

# ------------------------------
# Environment: 3x4 grid
# ------------------------------
class GridWorld3x4:
    def __init__(self, penalty_minus_one=True, step_cost=-0.04, seed=0):
        self.rows, self.cols = 3, 4
        self.start = (0,0)
        self.terminal_pos_plus  = (2,3)  # +1
        self.terminal_pos_minus = (1,3)  # -1 (or -200 if flag False)
        self.wall_pos = (1,1)            # impassable cell (gray)
        self.step_cost = step_cost
        self.penalty_minus_one_flag = penalty_minus_one
        self.rng = np.random.default_rng(seed)
        # action deltas: N,E,S,W (row increases upward)
        self.actions = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
        self.left_of  = {0:3, 1:0, 2:1, 3:2}
        self.right_of = {0:1, 1:2, 2:3, 3:0}
        self.nA = 4
        self.nS = self.rows * self.cols

    def is_terminal(self, s):
        return s in (self.terminal_pos_plus, self.terminal_pos_minus)

    def is_wall(self, s):
        return s == self.wall_pos

    def reset(self):
        return self.start

    def step(self, s, a):
        # If already terminal, stay with zero reward
        if self.is_terminal(s):
            return s, 0.0, True
        # Sample slip
        p = self.rng.random()
        if p < 0.8:
            a_exec = a
        elif p < 0.9:
            a_exec = self.left_of[a]
        else:
            a_exec = self.right_of[a]
        dr, dc = self.actions[a_exec]
        r, c = s[0] + dr, s[1] + dc
        # Check boundaries
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            r, c = s  # stay
        # Check wall collision: stay in place
        elif self.is_wall((r, c)):
            r, c = s
        s2 = (r, c)
        # Rewards
        if s2 == self.terminal_pos_plus:
            return s2, 1.0, True
        if s2 == self.terminal_pos_minus:
            return s2, (-1.0 if self.penalty_minus_one_flag else -200.0), True
        return s2, self.step_cost, False

    def to_index(self, s):
        return s[0] * self.cols + s[1]

    def from_index(self, i):
        return (i // self.cols, i % self.cols)

# ------------------------------
# Replay buffer
# ------------------------------
Transition = namedtuple('Transition', ['s', 'a', 'r', 's2', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # convert to tensors later
        return batch

    def __len__(self):
        return len(self.buffer)

# ------------------------------
# Network
# ------------------------------
class QNetwork(nn.Module):
    def __init__(self, n_states=12, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# Hyperparams dataclass
# ------------------------------
@dataclass
class DQNHyper:
    episodes: int = 4000
    max_steps: int = 100
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 5000
    start_train_after: int = 200  # number of transitions before starting training
    target_update_every: int = 100  # gradient steps between target network copy
    hidden_size: int = 64
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2500
    seed: int = 123
    eval_frequency: int = 200  # evaluate every N episodes
    eval_episodes: int = 50  # number of episodes for evaluation

# ------------------------------
# Utilities
# ------------------------------
def one_hot_state(s_idx, nS):
    v = np.zeros(nS, dtype=np.float32)
    v[s_idx] = 1.0
    return v

def linear_eps(ep, start, end, decay_episodes):
    frac = max(0.0, min(1.0, 1.0 - ep / float(max(1, decay_episodes))))
    return float(end + frac * (start - end))

# ------------------------------
# DQN training loop
# ------------------------------
def train_dqn(env: GridWorld3x4, hp: DQNHyper, verbose=True):
    # seeds
    random.seed(hp.seed); np.random.seed(hp.seed); torch.manual_seed(hp.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nS, nA = env.nS, env.nA
    policy_net = QNetwork(n_states=nS, n_actions=nA, hidden=hp.hidden_size).to(device)
    target_net = QNetwork(n_states=nS, n_actions=nA, hidden=hp.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optim = torch.optim.Adam(policy_net.parameters(), lr=hp.lr)
    replay = ReplayBuffer(hp.buffer_size)

    returns = []
    steps_hist = []
    eps_hist = []
    loss_hist = []
    eval_returns = []
    eval_steps = []
    eval_episodes = []
    grad_steps = 0

    for ep in range(hp.episodes):
        eps = linear_eps(ep, hp.eps_start, hp.eps_end, hp.eps_decay_episodes)
        eps_hist.append(eps)
        s = env.reset()
        s_idx = env.to_index(s)
        G = 0.0
        disc = 1.0
        done = False
        steps = 0

        for t in range(hp.max_steps):
            steps += 1
            # get action ε-greedy
            if random.random() < eps:
                a = random.randrange(nA)
            else:
                with torch.no_grad():
                    st = torch.tensor(one_hot_state(s_idx, nS), device=device).unsqueeze(0)
                    qvals = policy_net(st)  # shape [1, nA]
                    a = int(torch.argmax(qvals, dim=1).item())

            s2, r, done = env.step(s, a)
            s2_idx = env.to_index(s2)
            replay.push(s_idx, a, r, s2_idx, done)

            G += disc * r
            disc *= hp.gamma

            s = s2
            s_idx = s2_idx

            # start training after we have some transitions
            if len(replay) >= hp.start_train_after:
                batch = replay.sample(min(hp.batch_size, len(replay)))
                # build tensors
                s_batch = torch.tensor([one_hot_state(x.s, nS) for x in batch], device=device)
                a_batch = torch.tensor([x.a for x in batch], dtype=torch.long, device=device).unsqueeze(1)
                r_batch = torch.tensor([x.r for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
                s2_batch = torch.tensor([one_hot_state(x.s2, nS) for x in batch], device=device)
                done_batch = torch.tensor([0.0 if x.done else 1.0 for x in batch], dtype=torch.float32, device=device).unsqueeze(1)

                # Q(s,a)
                q_values = policy_net(s_batch).gather(1, a_batch)  # shape [B,1]
                # target: r + gamma * max_a' Q_target(s', a') * (not done)
                with torch.no_grad():
                    q_next = target_net(s2_batch)
                    q_next_max, _ = torch.max(q_next, dim=1, keepdim=True)
                    target = r_batch + hp.gamma * q_next_max * done_batch

                loss = nn.functional.mse_loss(q_values, target)
                optim.zero_grad()
                loss.backward()
                # gradient clipping (optional)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optim.step()

                loss_hist.append(loss.item())
                grad_steps += 1

                # update target network periodically
                if grad_steps % hp.target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        returns.append(G)
        steps_hist.append(steps)

        # Evaluation
        if ep % hp.eval_frequency == 0 or ep == hp.episodes - 1:
            eval_ret, eval_step = evaluate_policy(env, policy_net, num_episodes=hp.eval_episodes, gamma=hp.gamma, max_steps=hp.max_steps)
            eval_returns.append(np.mean(eval_ret))
            eval_steps.append(np.mean(eval_step))
            eval_episodes.append(ep)

        if verbose and (ep % 200 == 0 or ep == hp.episodes - 1):
            avg50 = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            eval_info = ""
            if len(eval_returns) > 0:
                eval_info = f"  eval_avg={eval_returns[-1]:.3f}"
            print(f"Ep {ep:4d}/{hp.episodes}  return={G:.3f}  avg50={avg50:.3f}  eps={eps:.3f}  replay={len(replay)}{eval_info}")

    # final copy
    target_net.load_state_dict(policy_net.state_dict())

    stats = {
        'policy_net': policy_net,
        'target_net': target_net,
        'returns': np.array(returns),
        'steps_hist': np.array(steps_hist),
        'eps_hist': np.array(eps_hist),
        'loss_hist': np.array(loss_hist),
        'eval_returns': np.array(eval_returns),
        'eval_steps': np.array(eval_steps),
        'eval_episodes': np.array(eval_episodes),
    }
    return stats

# ------------------------------
# Utilities for evaluation and plotting
# ------------------------------
def derive_V_and_policy_from_net(env, net, device=None):
    nS, nA = env.nS, env.nA
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    net.eval()
    V = np.zeros((env.rows, env.cols), dtype=float)
    Pi = np.full((env.rows, env.cols), -1, dtype=int)
    with torch.no_grad():
        for i in range(nS):
            s = env.from_index(i)
            if env.is_terminal(s):
                V[s[0], s[1]] = 0.0
                Pi[s[0], s[1]] = -1
            else:
                x = torch.tensor(one_hot_state(i, nS), dtype=torch.float32, device=device).unsqueeze(0)
                q = net(x).cpu().numpy().reshape(-1)
                # If wall: set to nan/ignore (we won't step into it anyway)
                if env.is_wall(s):
                    V[s[0], s[1]] = np.nan
                    Pi[s[0], s[1]] = -1
                else:
                    V[s[0], s[1]] = float(np.max(q))
                    Pi[s[0], s[1]] = int(np.argmax(q))
    return V, Pi

def policy_arrows(Pi):
    sym = {0:'↑',1:'→',2:'↓',3:'←',-1:'T'}
    lines=[]
    for r in reversed(range(Pi.shape[0])):
        lines.append(" ".join(sym[int(a)] for a in Pi[r]))
    return "\n".join(lines)

def evaluate_policy(env, net, num_episodes=100, gamma=0.99, max_steps=100, seed=42):
    """Evaluate the trained policy over multiple episodes"""
    rng = np.random.default_rng(seed)
    old_rng = env.rng
    env.rng = rng
    device = next(net.parameters()).device
    net.eval()
    
    eval_returns = []
    eval_steps = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            s = env.reset()
            s_idx = env.to_index(s)
            G = 0.0
            disc = 1.0
            steps = 0
            
            for _ in range(max_steps):
                steps += 1
                x = torch.tensor(one_hot_state(s_idx, env.nS), dtype=torch.float32, device=device).unsqueeze(0)
                q = net(x)
                a = int(torch.argmax(q, dim=1).item())
                
                s2, r, done = env.step(s, a)
                s2_idx = env.to_index(s2)
                
                G += disc * r
                disc *= gamma
                
                s = s2
                s_idx = s2_idx
                
                if done:
                    break
            
            eval_returns.append(G)
            eval_steps.append(steps)
    
    env.rng = old_rng
    return np.array(eval_returns), np.array(eval_steps)

def simulate_greedy_trajectory(env, net, gamma=0.99, max_steps=50, seed=999):
    rng = np.random.default_rng(seed)
    old_rng = env.rng
    env.rng = rng
    s = env.reset()
    states = [s]
    rewards = []
    device = next(net.parameters()).device
    for _ in range(max_steps):
        i = env.to_index(s)
        if env.is_terminal(s):
            break
        x = torch.tensor(one_hot_state(i, env.nS), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = net(x)
            a = int(torch.argmax(q, dim=1).item())
        s2, r, done = env.step(s, a)
        states.append(s2)
        rewards.append(r)
        s = s2
        if done: break
    env.rng = old_rng
    # discounted return
    G = 0.0
    disc = 1.0
    for r in rewards:
        G += disc * r
        disc *= gamma
    return states, rewards, G


# ------------------------------
# Main: train and show results
# ------------------------------
def main():
    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"dqn_results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving results to folder: {output_folder}")
    
    hp = DQNHyper(
        episodes=4000,
        max_steps=100,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=5000,
        start_train_after=200,
        target_update_every=200,
        hidden_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=2500,
        seed=123,
        eval_frequency=200,
        eval_episodes=50
    )

    env = GridWorld3x4(penalty_minus_one=True, step_cost=-0.04, seed=hp.seed)
    stats = train_dqn(env, hp, verbose=True)
    net = stats['policy_net']

    # Plots: returns
    returns = stats['returns']
    steps_hist = stats['steps_hist']
    eps_hist = stats['eps_hist']
    loss_hist = stats['loss_hist']
    eval_returns = stats['eval_returns']
    eval_steps = stats['eval_steps']
    eval_episodes = stats['eval_episodes']

    # Training and evaluation learning curves
    plt.figure(figsize=(12, 8))
    
    # Training returns
    plt.subplot(2, 2, 1)
    plt.plot(returns, label='Training Return/episode', alpha=0.7)
    if len(returns) >= 50:
        ma = np.convolve(returns, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(returns)), ma, label='Training MA(50)', linewidth=2)
    
    # Evaluation returns
    if len(eval_returns) > 0:
        plt.plot(eval_episodes, eval_returns, 'o-', label='Evaluation Return', color='red', linewidth=2, markersize=4)
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Learning Curves: Training vs Evaluation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(steps_hist, label='Training Steps/episode', alpha=0.7)
    if len(steps_hist) >= 50:
        ma = np.convolve(steps_hist, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(steps_hist)), ma, label='Training MA(50)', linewidth=2)
    
    if len(eval_steps) > 0:
        plt.plot(eval_episodes, eval_steps, 'o-', label='Evaluation Steps', color='red', linewidth=2, markersize=4)
    
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Length: Training vs Evaluation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Epsilon schedule
    plt.subplot(2, 2, 3)
    plt.plot(eps_hist)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Schedule')
    plt.grid(True, alpha=0.3)

    # Training loss
    plt.subplot(2, 2, 4)
    if len(loss_hist) > 0:
        # smooth loss with moving avg if many points
        w = min(200, max(1, len(loss_hist)//20))
        if len(loss_hist) >= w:
            ma = np.convolve(loss_hist, np.ones(w)/w, mode='valid')
            plt.plot(range(len(ma)), ma)
        else:
            plt.plot(loss_hist)
        plt.xlabel('Gradient step')
        plt.ylabel('MSE loss')
        plt.title('Training Loss (smoothed)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {output_folder}/learning_curves.png")

    # Value heatmap and greedy policy
    V, Pi = derive_V_and_policy_from_net(env, net)
    plt.figure(figsize=(8, 6))
    plt.imshow(V, origin='lower')
    plt.colorbar()
    plt.title('State values V(s) = max_a Q(s,a) (DQN estimate)')
    plt.xticks(range(env.cols), [str(c+1) for c in range(env.cols)])
    plt.yticks(range(env.rows), [str(r+1) for r in range(env.rows)])
    plt.savefig(f'{output_folder}/value_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved value heatmap to {output_folder}/value_heatmap.png")

    print("Final greedy policy (top row first; arrows N/E/S/W; T=terminal):")
    print(policy_arrows(Pi))

    # Greedy trajectory after training
    states, rewards, G = simulate_greedy_trajectory(env, net, gamma=hp.gamma, max_steps=50, seed=999)
    print("\nGreedy trajectory states:", states)
    print("Greedy trajectory rewards:", rewards)
    print(f"Total discounted return of this rollout (γ={hp.gamma}): {G:.3f}")
    
    # Save trajectory plot
    plt.figure(figsize=(8, 6))
    for r in range(env.rows+1): plt.plot([0, env.cols], [r, r], color='k', linewidth=0.5)
    for c in range(env.cols+1): plt.plot([c, c], [0, env.rows], color='k', linewidth=0.5)
    xs = [c+0.5 for (r,c) in states]
    ys = [r+0.5 for (r,c) in states]
    plt.plot(xs, ys, marker='o')
    s = env.start
    plt.text(s[1]+0.5, s[0]+0.5, "START", ha='center', va='center')
    tp = env.terminal_pos_plus; tm = env.terminal_pos_minus
    plt.text(tp[1]+0.5, tp[0]+0.5, "+1", ha='center', va='center')
    plt.text(tm[1]+0.5, tm[0]+0.5, "-1", ha='center', va='center')
    plt.gca().set_xlim(0, env.cols); plt.gca().set_ylim(0, env.rows)
    plt.gca().set_aspect('equal'); plt.title(f"Greedy trajectory; discounted return = {G:.3f}")
    plt.xlabel("col"); plt.ylabel("row")
    plt.savefig(f'{output_folder}/greedy_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved greedy trajectory to {output_folder}/greedy_trajectory.png")

    # Show Q-values for each state
    rows = []
    device = next(net.parameters()).device
    for r in range(env.rows):
        for c in range(env.cols):
            i = r*env.cols + c
            s = (r,c)
            if env.is_wall(s):
                rows.append({"row":r+1,"col":c+1,"Q_N":np.nan,"Q_E":np.nan,"Q_S":np.nan,"Q_W":np.nan,"V":np.nan,"best_a_idx":-1})
                continue
            x = torch.tensor(one_hot_state(i, env.nS), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = net(x).cpu().numpy().reshape(-1)
            rows.append({"row":r+1,"col":c+1,"Q_N":q[0],"Q_E":q[1],"Q_S":q[2],"Q_W":q[3],"V":float(np.max(q)),"best_a_idx":int(np.argmax(q))})
    df = pd.DataFrame(rows)
    print("\nQ-table (approx) per state:\n", df)
    df.to_csv(f"{output_folder}/dqn_q_table.csv", index=False)
    print(f"\nSaved approximated Q-table to {output_folder}/dqn_q_table.csv")
    
    # Save training statistics
    training_stats = {
        'final_training_return': np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns),
        'final_evaluation_return': eval_returns[-1] if len(eval_returns) > 0 else None,
        'total_episodes': len(returns),
        'total_gradient_steps': len(loss_hist),
        'final_epsilon': eps_hist[-1],
        'hyperparameters': {
            'episodes': hp.episodes,
            'max_steps': hp.max_steps,
            'gamma': hp.gamma,
            'lr': hp.lr,
            'batch_size': hp.batch_size,
            'buffer_size': hp.buffer_size,
            'hidden_size': hp.hidden_size,
            'eps_start': hp.eps_start,
            'eps_end': hp.eps_end,
            'eps_decay_episodes': hp.eps_decay_episodes,
            'eval_frequency': hp.eval_frequency,
            'eval_episodes': hp.eval_episodes,
            'seed': hp.seed
        }
    }
    
    # Save training statistics to file
    import json
    with open(f"{output_folder}/training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"Saved training statistics to {output_folder}/training_stats.json")
    
    print(f"\nAll results saved to folder: {output_folder}")
    print("Files created:")
    print(f"  - {output_folder}/learning_curves.png")
    print(f"  - {output_folder}/value_heatmap.png") 
    print(f"  - {output_folder}/greedy_trajectory.png")
    print(f"  - {output_folder}/dqn_q_table.csv")
    print(f"  - {output_folder}/training_stats.json")

if __name__ == "__main__":
    main()
