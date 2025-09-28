# Complete, runnable Q-learning demo for the 3x4 stochastic gridworld.
# It trains a Q-table, then shows:
# 1) Learning curve (returns)
# 2) Episode length curve (optional but useful)
# 3) Exploration (epsilon) schedule
# 4) Value heatmap & final greedy policy
# 5) A visualization of the GREEDY TRAJECTORY from START after training, with total return
#
# You can tweak HYPERPARAMS below and re-run.

import numpy as np, random, matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd

# ------------------------------
# Environment
# ------------------------------
class GridWorld3x4:
    """
    3 rows x 4 columns grid.
    0-indexed coordinates: (row, col) with row=0 at bottom, row=2 at top.
    START = (0,0). Terminals: +1 at (2,3), -1 at (1,3) (can be -200 via flag).
    Actions: 0:N, 1:E, 2:S, 3:W.
    Stochastic: intended 0.8, left 0.1, right 0.1. Walls keep you in place.
    Non-terminal step reward = -0.04.
    """
    def __init__(self, penalty_minus_one=True, step_cost=-0.04, seed=0):
        self.rows, self.cols = 3, 4
        self.start = (0,0)
        self.terminal_pos_plus  = (2,3)  # +1
        self.terminal_pos_minus = (1,3)  # -1
        self.wall_pos = (1, 1)  # Wall at position (1,1) - impassable
        self.step_cost = step_cost
        self.penalty_minus_one_flag = penalty_minus_one
        self.rng = np.random.default_rng(seed)
        # action deltas: N,E,S,W (row increases upward)
        self.actions = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
        self.left_of  = {0:3, 1:0, 2:1, 3:2}
        self.right_of = {0:1, 1:2, 2:3, 3:0}

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
        if p < 0.8:   a_exec = a
        elif p < 0.9: a_exec = self.left_of[a]
        else:         a_exec = self.right_of[a]
        dr, dc = self.actions[a_exec]
        r, c = s[0] + dr, s[1] + dc
        # Check boundaries
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            r, c = s
        # Check wall collision
        elif self.is_wall((r, c)):
            r, c = s  # Stay in current position if hitting wall
        s2 = (r, c)
        # Rewards
        if s2 == self.terminal_pos_plus:
            return s2, 1.0, True
        if s2 == self.terminal_pos_minus:
            return s2, (-1.0 if self.penalty_minus_one_flag else -200.0), True
        return s2, self.step_cost, False

# ------------------------------
# Q-learning core
# ------------------------------
@dataclass
class HyperParams:
    episodes: int = 4000
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2500
    max_steps: int = 100
    seed: int = 123

def run_q_learning(env: GridWorld3x4, hp: HyperParams):
    nS, nA = env.rows * env.cols, 4
    def idx(s): return s[0]*env.cols + s[1]
    Q = np.zeros((nS, nA), dtype=float)
    returns, steps_hist, eps_hist = [], [], []

    random.seed(hp.seed)
    np.random.seed(hp.seed)

    for ep in range(hp.episodes):
        # Linear ε decay
        eps = hp.eps_end + max(0, hp.eps_decay_episodes-ep)*(hp.eps_start-hp.eps_end)/max(1, hp.eps_decay_episodes)
        eps = max(hp.eps_end, min(hp.eps_start, eps))
        eps_hist.append(eps)

        s = env.reset()
        G, disc = 0.0, 1.0
        for t in range(hp.max_steps):
            si = idx(s)
            # ε-greedy action
            if random.random() < eps:
                a = random.randint(0, nA-1)
            else:
                m = np.max(Q[si]); best = np.flatnonzero(Q[si]==m)
                a = int(np.random.choice(best))
            s2, r, done = env.step(s, a)
            sj = idx(s2)
            # Q-learning target (off-policy, bootstrap with max)
            target = r + (0.0 if env.is_terminal(s2) else hp.gamma*np.max(Q[sj]))
            Q[si, a] += hp.alpha * (target - Q[si, a])
            # book-keeping
            G += disc * r
            disc *= hp.gamma
            s = s2
            if done:
                steps_hist.append(t+1)
                break
        else:
            steps_hist.append(hp.max_steps)
        returns.append(G)
    return Q, returns, steps_hist, eps_hist

def derive_V_and_policy(env, Q):
    V = np.zeros((env.rows, env.cols))
    Pi = np.full((env.rows, env.cols), -1, dtype=int)
    for r in range(env.rows):
        for c in range(env.cols):
            i = r*env.cols + c
            if env.is_terminal((r,c)):
                V[r,c] = 0.0
                Pi[r,c] = -1
            else:
                V[r,c] = float(np.max(Q[i]))
                Pi[r,c] = int(np.argmax(Q[i]))
    return V, Pi

def policy_arrows(Pi):
    sym = {0:'↑',1:'→',2:'↓',3:'←',-1:'T'}
    lines=[]
    for r in reversed(range(Pi.shape[0])):
        lines.append(" ".join(sym[int(a)] for a in Pi[r]))
    return "\n".join(lines)

# Simulate one GREEDY trajectory after training, compute return, and plot the path
def simulate_greedy_trajectory(env: GridWorld3x4, Q, gamma=0.99, max_steps=50, seed=999):
    rng = np.random.default_rng(seed)
    # temporarily swap env RNG to make the rollout reproducible
    old_rng = env.rng
    env.rng = rng
    s = env.reset()
    states = [s]
    rewards = []
    nA = 4
    def idx(s): return s[0]*env.cols + s[1]
    for t in range(max_steps):
        i = idx(s)
        a = int(np.argmax(Q[i]))  # greedy
        s2, r, done = env.step(s, a)
        states.append(s2); rewards.append(r)
        s = s2
        if done: break
    # restore RNG
    env.rng = old_rng
    # compute discounted return of this rollout
    G = 0.0
    disc = 1.0
    for r in rewards:
        G += disc * r
        disc *= gamma
    return states, rewards, G

def plot_trajectory(env: GridWorld3x4, states, title="Greedy trajectory after training"):
    # Draw grid and connect centers of visited cells
    plt.figure()
    # grid lines
    for r in range(env.rows+1):
        plt.plot([0, env.cols], [r, r])
    for c in range(env.cols+1):
        plt.plot([c, c], [0, env.rows])
    # cell centers
    xs = [c+0.5 for (r,c) in states]
    ys = [r+0.5 for (r,c) in states]
    # path
    plt.plot(xs, ys, marker='o')
    # mark start and terminals
    s = env.start
    plt.text(s[1]+0.5, s[0]+0.5, "START", ha='center', va='center')
    tp = env.terminal_pos_plus
    tm = env.terminal_pos_minus
    plt.text(tp[1]+0.5, tp[0]+0.5, "+1", ha='center', va='center')
    plt.text(tm[1]+0.5, tm[0]+0.5, "-1", ha='center', va='center')
    plt.gca().set_xlim(0, env.cols)
    plt.gca().set_ylim(0, env.rows)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("col"); plt.ylabel("row")
    plt.show()

# ------------------------------
# HYPERPARAMS and Training
# ------------------------------
hp = HyperParams(
    episodes=4000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=2500,
    max_steps=100,
    seed=123
)

env = GridWorld3x4(penalty_minus_one=True, step_cost=-0.04, seed=hp.seed)
Q, returns, steps_hist, eps_hist = run_q_learning(env, hp)
V, Pi = derive_V_and_policy(env, Q)

# ------------------------------
# 1) Learning curve (returns)
# ------------------------------
plt.figure()
plt.plot(returns, label="Return/episode")
if len(returns) >= 50:
    ma = np.convolve(returns, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(returns)), ma, label="Moving avg (50)")
plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Learning curve"); plt.legend()
plt.show()

# 2) Episode length (optional but helpful)
plt.figure()
plt.plot(steps_hist, label="Steps/episode")
if len(steps_hist) >= 50:
    ma = np.convolve(steps_hist, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(steps_hist)), ma, label="Moving avg (50)")
plt.xlabel("Episode"); plt.ylabel("Steps"); plt.title("Episode length"); plt.legend()
plt.show()

# 3) Exploration schedule
plt.figure()
plt.plot(eps_hist)
plt.xlabel("Episode"); plt.ylabel("epsilon"); plt.title("Exploration schedule")
plt.show()

# 4) Value heatmap
plt.figure()
plt.imshow(V, origin='lower')
plt.colorbar()
plt.title("State values V(s) = max_a Q(s,a)")
plt.xticks(range(env.cols), [str(c+1) for c in range(env.cols)])
plt.yticks(range(env.rows), [str(r+1) for r in range(env.rows)])
plt.show()

# 5) Final greedy policy (text)
print("Final greedy policy (top row first; arrows N/E/S/W; T=terminal):")
print(policy_arrows(Pi))

# 6) Greedy trajectory after training, with return
states, rewards, G = simulate_greedy_trajectory(env, Q, gamma=hp.gamma, max_steps=50, seed=999)
print("\nGreedy trajectory states:", states)
print("Greedy trajectory rewards:", rewards)
print(f"Total discounted return of this rollout (γ={hp.gamma}): {G:.3f}")
plot_trajectory(env, states, title=f"Greedy trajectory; discounted return = {G:.3f}")

# 7) Show the Q-table and best action per state
rows=[]
for r in range(env.rows):
    for c in range(env.cols):
        i = r*env.cols + c
        qN,qE,qS,qW = Q[i]
        rows.append({"row":r+1,"col":c+1,"Q_N":qN,"Q_E":qE,"Q_S":qS,"Q_W":qW,
                     "V=max(Q)":max(Q[i]),"best_a_idx":int(np.argmax(Q[i]))})
df = pd.DataFrame(rows)


