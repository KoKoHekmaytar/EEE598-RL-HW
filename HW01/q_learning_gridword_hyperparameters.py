import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# ==============================
# Environment: 3x4 GridWorld
# ==============================
class GridWorld3x4:
    def __init__(self, penalty_minus_one=True, step_cost=-0.04, seed=0):
        self.rows, self.cols = 3, 4
        self.start = (0, 0)
        self.terminal_pos_plus  = (2, 3)  # +1
        self.terminal_pos_minus = (1, 3)  # -1
        self.wall_pos = (1, 1)  # Wall at position (1,1) - impassable
        self.step_cost = step_cost
        self.penalty_minus_one_flag = penalty_minus_one
        self.rng = np.random.default_rng(seed)

        # Actions: 0=N, 1=E, 2=S, 3=W (row increases upward here)
        self.actions = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        self.left_of  = {0: 3, 1: 0, 2: 1, 3: 2}
        self.right_of = {0: 1, 1: 2, 2: 3, 3: 0}

    def is_terminal(self, s):
        return s in (self.terminal_pos_plus, self.terminal_pos_minus)
    
    def is_wall(self, s):
        return s == self.wall_pos

    def reset(self):
        return self.start

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0.0, True

        # Slip dynamics
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


# ==============================
# Q-learning core
# ==============================
@dataclass
class HyperParams:
    episodes: int = 3000
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 1500  # for 'linear'
    eps_half_life: int = 400        # for 'exp' (episodes to halve (start-end) gap)
    eps_schedule: str = "linear"    # linear | exp | constant
    max_steps: int = 100
    seed: int = 123


def make_epsilon_schedule(hp: HyperParams):
    sched = hp.eps_schedule.lower()
    if sched == "constant":
        def eps_fn(ep): return hp.eps_start
        return eps_fn
    if sched == "exp":
        lam = np.log(2.0) / float(max(1, hp.eps_half_life))  # decay constant
        def eps_fn(ep):
            gap = (hp.eps_start - hp.eps_end) * np.exp(-lam * ep)
            return float(hp.eps_end + max(0.0, gap))
        return eps_fn
    # linear (default)
    def eps_fn(ep):
        frac = max(0.0, min(1.0, 1.0 - ep / float(max(1, hp.eps_decay_episodes))))
        return float(hp.eps_end + frac * (hp.eps_start - hp.eps_end))
    return eps_fn


def run_q_learning(env: GridWorld3x4, hp: HyperParams):
    nS, nA = env.rows * env.cols, 4
    def idx(s): return s[0] * env.cols + s[1]
    Q = np.zeros((nS, nA), dtype=float)
    returns = []

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    eps_fn = make_epsilon_schedule(hp)

    for ep in range(hp.episodes):
        s = env.reset()
        G, disc = 0.0, 1.0
        eps = eps_fn(ep)

        for _ in range(hp.max_steps):
            si = idx(s)

            # ε-greedy
            if random.random() < eps:
                a = random.randint(0, nA - 1)
            else:
                m = np.max(Q[si])
                best = np.flatnonzero(Q[si] == m)
                a = int(np.random.choice(best))

            s2, r, done = env.step(s, a)
            sj = idx(s2)

            # Q-learning target
            target = r + (0.0 if env.is_terminal(s2) else hp.gamma * np.max(Q[sj]))
            Q[si, a] += hp.alpha * (target - Q[si, a])

            # Book-keeping for the episode return
            G += disc * r
            disc *= hp.gamma

            s = s2
            if done:
                break

        returns.append(G)

    return np.array(returns)


def moving_avg(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')


# ==============================
# Sweeps & Plots
# ==============================
def main():
    # Create output directory for plots
    output_dir = "q_learning_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ----- Sweep α (learning rate) -----
    alpha_list = [0.05, 0.1, 0.3, 0.5]
    episodes = 3000
    plt.figure(figsize=(10, 6))
    for a in alpha_list:
        hp = HyperParams(
            episodes=episodes,
            alpha=a,
            gamma=0.99,
            eps_schedule="linear",
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_episodes=1500
        )
        env = GridWorld3x4(penalty_minus_one=True, step_cost=-0.04, seed=hp.seed)
        rets = run_q_learning(env, hp)
        ma = moving_avg(rets, w=50)
        xs = range(len(ma)) if len(ma) > 0 else range(len(rets))
        plt.plot(xs, ma if len(ma) > 0 else rets, label=f"alpha={a}")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return vs Episode — varying learning rate α")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alpha_sweep.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Alpha sweep plot saved to {output_dir}/alpha_sweep.png")

    # ----- Sweep γ (discount) -----
    gamma_list = [0.8,0.85, 0.9,0.95, 0.99]
    plt.figure(figsize=(10, 6))
    for g in gamma_list:
        hp = HyperParams(
            episodes=episodes,
            alpha=0.1,
            gamma=g,
            eps_schedule="linear",
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_episodes=1500
        )
        env = GridWorld3x4(penalty_minus_one=True, step_cost=-0.04, seed=hp.seed)
        rets = run_q_learning(env, hp)
        ma = moving_avg(rets, w=50)
        xs = range(len(ma)) if len(ma) > 0 else range(len(rets))
        plt.plot(xs, ma if len(ma) > 0 else rets, label=f"gamma={g}")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return vs Episode — varying discount γ")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gamma_sweep.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Gamma sweep plot saved to {output_dir}/gamma_sweep.png")

    # ----- Sweep ε schedule -----
    print("Starting epsilon schedule sweep...")
    schedules = [
        dict(name="linear",   eps_schedule="linear",   eps_start=1.0, eps_end=0.05, eps_decay_episodes=1500),
        dict(name="exp",      eps_schedule="exp",      eps_start=1.0, eps_end=0.05, eps_half_life=400),
        dict(name="constant", eps_schedule="constant", eps_start=0.1, eps_end=0.1),
    ]
    plt.figure(figsize=(10, 6))
    for cfg in schedules:
        # Extract name before passing to HyperParams
        name = cfg.pop('name')
        hp = HyperParams(
            episodes=episodes,
            alpha=0.1,
            gamma=0.99,
            **cfg
        )
        env = GridWorld3x4(penalty_minus_one=True, step_cost=-0.04, seed=hp.seed)
        rets = run_q_learning(env, hp)
        ma = moving_avg(rets, w=50)
        xs = range(len(ma)) if len(ma) > 0 else range(len(rets))
        plt.plot(xs, ma if len(ma) > 0 else rets, label=f"ε: {name}")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return vs Episode — varying exploration schedule ε")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_schedule_sweep.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Epsilon schedule sweep plot saved to {output_dir}/epsilon_schedule_sweep.png")


if __name__ == "__main__":
    try:
        main()
        print("Q-learning hyperparameter sweep completed successfully!")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages: pip install numpy matplotlib")
    except Exception as e:
        print(f"Error running Q-learning: {e}")
        import traceback
        traceback.print_exc()
