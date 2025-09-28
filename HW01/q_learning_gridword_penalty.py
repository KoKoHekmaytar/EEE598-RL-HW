import numpy as np, random, matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# ------------------------------
# Environment
# ------------------------------
class GridWorld3x4:
    """
    3x4 grid; START at (row=0,col=0); +1 terminal at (2,3); trap at (1,3).
    Actions: 0=N, 1=E, 2=S, 3=W. Slip: intended 0.8, left 0.1, right 0.1.
    Step cost = -0.04.
    """
    def __init__(self, penalty_minus_one=True, step_cost=-0.04, seed=0):
        self.rows, self.cols = 3, 4
        self.start = (0,0)
        self.terminal_pos_plus  = (2,3)
        self.terminal_pos_minus = (1,3)
        self.wall_pos = (1,1)  # Wall at position (1,1) - impassable
        self.step_cost = step_cost
        self.penalty_minus_one_flag = penalty_minus_one   # True -> -1, False -> -200
        self.rng = np.random.default_rng(seed)
        self.actions = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}  # N,E,S,W
        self.left_of  = {0:3, 1:0, 2:1, 3:2}
        self.right_of = {0:1, 1:2, 2:3, 3:0}

    def is_terminal(self, s):
        return s in (self.terminal_pos_plus, self.terminal_pos_minus)
    
    def is_wall(self, s):
        return s == self.wall_pos

    def reset(self):
        return self.start

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0.0, True
        # slip
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
        if s2 == self.terminal_pos_plus:
            return s2, 1.0, True
        if s2 == self.terminal_pos_minus:
            return s2, (-1.0 if self.penalty_minus_one_flag else -1000.0), True  # Even more extreme penalty
        return s2, self.step_cost, False

# ------------------------------
# Q-learning
# ------------------------------
@dataclass
class HyperParams:
    episodes:int=6000  # More episodes for better convergence
    alpha:float=0.2    # Higher learning rate for faster learning
    gamma:float=0.85   # Lower discount factor - agent cares more about immediate rewards
    eps_start:float=1.0
    eps_end:float=0.01  # Lower final epsilon for more exploitation
    eps_decay_episodes:int=4000  # Longer decay period
    max_steps:int=100
    seed:int=42

def run_q_learning(env, hp:HyperParams):
    nS, nA = env.rows * env.cols, 4
    def idx(s): return s[0]*env.cols + s[1]
    Q = np.zeros((nS, nA), dtype=float)
    random.seed(hp.seed); np.random.seed(hp.seed)
    for ep in range(hp.episodes):
        # linear epsilon decay
        eps = hp.eps_end + max(0, hp.eps_decay_episodes-ep)*(hp.eps_start-hp.eps_end)/max(1, hp.eps_decay_episodes)
        eps = max(hp.eps_end, min(hp.eps_start, eps))
        s = env.reset()
        for _ in range(hp.max_steps):
            si = idx(s)
            # epsilon-greedy
            if random.random() < eps:
                a = random.randint(0, nA-1)
            else:
                m = np.max(Q[si]); best = np.flatnonzero(Q[si]==m)
                a = int(np.random.choice(best))
            s2, r, done = env.step(s, a)
            sj = idx(s2)
            target = r + (0.0 if env.is_terminal(s2) else hp.gamma*np.max(Q[sj]))
            Q[si, a] += hp.alpha * (target - Q[si, a])
            s = s2
            if done: break
    return Q

def value_and_policy(env, Q):
    V = np.zeros((env.rows, env.cols)); Pi = np.full((env.rows, env.cols), -1, int)
    for r in range(env.rows):
        for c in range(env.cols):
            if env.is_terminal((r,c)): 
                V[r,c]=0; Pi[r,c]=-1
            elif env.is_wall((r,c)):
                V[r,c]=-999; Pi[r,c]=-2  # Special value for wall
            else:
                i=r*env.cols+c; V[r,c]=np.max(Q[i]); Pi[r,c]=int(np.argmax(Q[i]))
    return V, Pi

def arrows(Pi):
    sym={0:'↑',1:'→',2:'↓',3:'←',-1:'T',-2:'█'}  # Added wall symbol
    lines=[]
    for r in reversed(range(Pi.shape[0])):
        lines.append(" ".join(sym[int(a)] for a in Pi[r]))
    return "\n".join(lines)

def plot_values(V, title, filename=None):
    plt.figure(figsize=(8, 6))
    # Create a masked array to handle the wall
    V_masked = np.ma.masked_where(V == -999, V)
    plt.imshow(V_masked, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(4), ["1","2","3","4"])
    plt.yticks(range(3), ["1","2","3"])
    # Add wall annotation
    plt.text(0.5, 2.5, "WALL", ha='center', va='center', fontsize=12, weight='bold', color='white')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    plt.show()

def greedy_rollout(env, Q, gamma=0.99, max_steps=50, seed=999):
    rng = np.random.default_rng(seed); old=env.rng; env.rng=rng
    s = env.reset(); states=[s]; G=0.0; disc=1.0
    def idx(s): return s[0]*env.cols + s[1]
    for _ in range(max_steps):
        a=int(np.argmax(Q[idx(s)]))
        s2,r,done = env.step(s,a)
        states.append(s2)
        G += disc*r; disc *= gamma
        s=s2
        if done: break
    env.rng=old
    return states, G

def plot_traj(env, states, title, filename=None):
    plt.figure(figsize=(8, 6))
    for r in range(env.rows+1): plt.plot([0, env.cols],[r,r], 'k-', linewidth=2)
    for c in range(env.cols+1): plt.plot([c,c],[0,env.rows], 'k-', linewidth=2)
    xs=[c+0.5 for (r,c) in states]; ys=[r+0.5 for (r,c) in states]
    plt.plot(xs, ys, marker='o', linewidth=3, markersize=8)
    s=env.start; plt.text(s[1]+0.5, s[0]+0.5, "START", ha='center', va='center', fontsize=12, weight='bold')
    tp=env.terminal_pos_plus; tm=env.terminal_pos_minus
    plt.text(tp[1]+0.5,tp[0]+0.5,"+1",ha='center',va='center', fontsize=12, weight='bold', color='green')
    plt.text(tm[1]+0.5,tm[0]+0.5,"TRAP",ha='center',va='center', fontsize=12, weight='bold', color='red')
    # Add wall visualization
    wall_r, wall_c = env.wall_pos
    plt.fill([wall_c, wall_c+1, wall_c+1, wall_c], [wall_r, wall_r, wall_r+1, wall_r+1], 
             color='gray', alpha=0.7, label='Wall')
    plt.text(wall_c+0.5, wall_r+0.5, "WALL", ha='center', va='center', fontsize=10, weight='bold', color='white')
    plt.gca().set_xlim(0, env.cols); plt.gca().set_ylim(0, env.rows); plt.gca().set_aspect('equal')
    plt.title(title); plt.xlabel("col"); plt.ylabel("row")
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot: {filename}")
    plt.show()

# ------------------------------
# Run both cases with the same seed/hyperparams
# ------------------------------
# Create output directory
output_dir = "trap_penalty_comparison"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

hp = HyperParams(episodes=8000, alpha=0.1, gamma=0.2, seed=456)  # Different seed
print(f"Hyperparameters: episodes={hp.episodes}, alpha={hp.alpha}, gamma={hp.gamma}")
print(f"Step cost: -0.2")  # Higher step cost to make paths more different

# Case A: trap = -1
print("\n" + "="*50)
print("CASE A: Trap penalty = -1")
print("="*50)
env_A = GridWorld3x4(penalty_minus_one=True,  step_cost=-0.2, seed=hp.seed)
Q_A = run_q_learning(env_A, hp)
V_A, Pi_A = value_and_policy(env_A, Q_A)
plot_values(V_A, "State values with trap = -1", os.path.join(output_dir, "values_trap_minus1.png"))
print("Greedy policy (trap = -1):\n" + arrows(Pi_A))
states_A, G_A = greedy_rollout(env_A, Q_A, gamma=hp.gamma)
plot_traj(env_A, states_A, f"Greedy trajectory (trap=-1); discounted return = {G_A:.3f}", 
          os.path.join(output_dir, "trajectory_trap_minus1.png"))

# Print Q-values for key states
print(f"\nQ-values for key states (trap = -1):")
print(f"State (0,0) - Start: {Q_A[0]}")
print(f"State (0,1) - Right of start: {Q_A[1]}")
print(f"State (1,1) - Upper path: {Q_A[5]}")
print(f"State (1,2) - Upper path: {Q_A[6]}")
print(f"State (0,2) - Lower path: {Q_A[2]}")
print(f"State (0,3) - Lower path: {Q_A[3]}")

# Case B: trap = -200
print("\n" + "="*50)
print("CASE B: Trap penalty = -200")
print("="*50)
env_B = GridWorld3x4(penalty_minus_one=False, step_cost=-0.2, seed=hp.seed)
Q_B = run_q_learning(env_B, hp)
V_B, Pi_B = value_and_policy(env_B, Q_B)
plot_values(V_B, "State values with trap = -200", os.path.join(output_dir, "values_trap_minus200.png"))
print("Greedy policy (trap = -200):\n" + arrows(Pi_B))
states_B, G_B = greedy_rollout(env_B, Q_B, gamma=hp.gamma)
plot_traj(env_B, states_B, f"Greedy trajectory (trap=-200); discounted return = {G_B:.3f}", 
          os.path.join(output_dir, "trajectory_trap_minus200.png"))

# Print Q-values for key states
print(f"\nQ-values for key states (trap = -200):")
print(f"State (0,0) - Start: {Q_B[0]}")
print(f"State (0,1) - Right of start: {Q_B[1]}")
print(f"State (1,1) - Upper path: {Q_B[5]}")
print(f"State (1,2) - Upper path: {Q_B[6]}")
print(f"State (0,2) - Lower path: {Q_B[2]}")
print(f"State (0,3) - Lower path: {Q_B[3]}")

print(f"\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)
print(f"Trap = -1:  Return = {G_A:.3f}, Trajectory length = {len(states_A)}")
print(f"Trap = -200: Return = {G_B:.3f}, Trajectory length = {len(states_B)}")
print(f"Difference in return: {abs(G_A - G_B):.3f}")
print(f"All plots saved to: {output_dir}/")
