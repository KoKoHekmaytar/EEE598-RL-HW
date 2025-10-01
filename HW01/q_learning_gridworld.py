import numpy as np, random, matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.animation as animation

class GridWorld3x4:
    def __init__(self, penalty_minus_one=True, step_cost=-0.04, seed=0):
        self.rows, self.cols = 3, 4
        self.start = (0,0)
        self.terminal_pos_plus  = (2,3)
        self.terminal_pos_minus = (1,3)
        self.wall_pos = (1, 1)
        self.step_cost = step_cost
        self.penalty_minus_one_flag = penalty_minus_one
        self.rng = np.random.default_rng(seed)
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
        if self.is_terminal(s): 
            return s, 0.0, True
        p = self.rng.random()
        if p < 0.8:   a_exec = a
        elif p < 0.9: a_exec = self.left_of[a]
        else:         a_exec = self.right_of[a]
        dr, dc = self.actions[a_exec]
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            r, c = s
        elif self.is_wall((r, c)):
            r, c = s
        s2 = (r, c)
        if s2 == self.terminal_pos_plus:
            return s2, 1.0, True
        if s2 == self.terminal_pos_minus:
            return s2, (-1.0 if self.penalty_minus_one_flag else -200.0), True
        return s2, self.step_cost, False

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
        eps = hp.eps_end + max(0, hp.eps_decay_episodes-ep)*(hp.eps_start-hp.eps_end)/max(1, hp.eps_decay_episodes)
        eps = max(hp.eps_end, min(hp.eps_start, eps))
        eps_hist.append(eps)

        s = env.reset()
        G, disc = 0.0, 1.0
        for t in range(hp.max_steps):
            si = idx(s)
            if random.random() < eps:
                a = random.randint(0, nA-1)
            else:
                m = np.max(Q[si]); best = np.flatnonzero(Q[si]==m)
                a = int(np.random.choice(best))
            s2, r, done = env.step(s, a)
            sj = idx(s2)
            target = r + (0.0 if env.is_terminal(s2) else hp.gamma*np.max(Q[sj]))
            Q[si, a] += hp.alpha * (target - Q[si, a])
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

plt.figure()
plt.plot(returns, label="Return/episode")
if len(returns) >= 50:
    ma = np.convolve(returns, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(returns)), ma, label="Moving avg (50)")
plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Learning curve"); plt.legend()
plt.show()

plt.figure()
plt.imshow(V, origin='lower')
plt.colorbar()
plt.title("Q-values V(s) = max_a Q(s,a)")
plt.xticks(range(env.cols), [str(c+1) for c in range(env.cols)])
plt.yticks(range(env.rows), [str(r+1) for r in range(env.rows)])
plt.show()



def create_agent_video(env, Q, max_steps=20, seed=42, filename='agent_movement.mp4'):
    rng = np.random.default_rng(seed)
    old_rng = env.rng
    env.rng = rng
    
    s = env.reset()
    states = [s]
    actions = []
    rewards = []
    
    def idx(s): return s[0]*env.cols + s[1]
    
    for step in range(max_steps):
        i = idx(s)
        a = int(np.argmax(Q[i]))
        s2, r, done = env.step(s, a)
        
        states.append(s2)
        actions.append(a)
        rewards.append(r)
        
        if done:
            break
        s = s2
    
    env.rng = old_rng
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for r in range(env.rows+1):
        ax.plot([0, env.cols], [r, r], 'k-', linewidth=3)
    for c in range(env.cols+1):
        ax.plot([c, c], [0, env.rows], 'k-', linewidth=3)
    
    start_r, start_c = env.start
    ax.add_patch(plt.Rectangle((start_c, start_r), 1, 1, facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2))
    ax.text(start_c+0.5, start_r+0.5, "START", ha='center', va='center', fontsize=16, weight='bold', color='darkgreen')
    
    goal_r, goal_c = env.terminal_pos_plus
    ax.add_patch(plt.Rectangle((goal_c, goal_r), 1, 1, facecolor='lightblue', alpha=0.8, edgecolor='darkblue', linewidth=2))
    ax.text(goal_c+0.5, goal_r+0.5, "+1", ha='center', va='center', fontsize=16, weight='bold', color='darkblue')
    
    trap_r, trap_c = env.terminal_pos_minus
    ax.add_patch(plt.Rectangle((trap_c, trap_r), 1, 1, facecolor='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=2))
    ax.text(trap_c+0.5, trap_r+0.5, "-1", ha='center', va='center', fontsize=16, weight='bold', color='darkred')
    
    wall_r, wall_c = env.wall_pos
    ax.add_patch(plt.Rectangle((wall_c, wall_r), 1, 1, facecolor='gray', alpha=0.9, edgecolor='black', linewidth=2))
    ax.text(wall_c+0.5, wall_r+0.5, "WALL", ha='center', va='center', fontsize=14, weight='bold', color='white')
    
    ax.set_xlim(-0.1, env.cols+0.1)
    ax.set_ylim(-0.1, env.rows+0.1)
    ax.set_aspect('equal')
    ax.set_title('Q-Learning Agent Movement Animation', fontsize=18, weight='bold', pad=20)
    ax.set_xlabel('Column', fontsize=14)
    ax.set_ylabel('Row', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    agent_marker, = ax.plot([], [], 'ro', markersize=20, label='Agent', markeredgecolor='darkred', markeredgewidth=2)
    path_line, = ax.plot([], [], 'b-', linewidth=4, alpha=0.8, label='Path')
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    def animate(frame):
        if frame < len(states):
            current_state = states[frame]
            x, y = current_state[1] + 0.5, current_state[0] + 0.5
            
            agent_marker.set_data([x], [y])
            
            if frame > 0:
                path_xs = [s[1] + 0.5 for s in states[:frame+1]]
                path_ys = [s[0] + 0.5 for s in states[:frame+1]]
                path_line.set_data(path_xs, path_ys)
            
            if frame < len(actions):
                action_names = {0: 'North ↑', 1: 'East →', 2: 'South ↓', 3: 'West ←'}
                action_name = action_names[actions[frame]]
                reward = rewards[frame]
                step_text.set_text(f'Step {frame+1}: {states[frame]} → {states[frame+1]}\nAction: {action_name}\nReward: {reward:.3f}')
            else:
                step_text.set_text(f'Episode Complete!\nTotal Steps: {len(states)-1}\nTotal Reward: {sum(rewards):.3f}')
        
        return agent_marker, path_line, step_text
    
    anim = animation.FuncAnimation(fig, animate, frames=len(states), 
                                 interval=800, repeat=False, blit=False)
    
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    print(f"Creating high-quality video: {filename}")
    gif_filename = filename.replace('.mp4', '.gif')
    anim.save(gif_filename, writer='pillow', fps=1.25)
    print(f"GIF saved as '{gif_filename}'")
    
    return states, actions, rewards, anim

print("Creating animated GIF...")
states, actions, rewards, anim = create_agent_video(env, Q, max_steps=20, seed=42, filename='q_learning_agent.mp4')
