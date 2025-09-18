"""
Enhanced GridWorld Navigation: ANN vs SNN with Comprehensive Metrics Tracking
Saves efficiency plots, rewards, parameter counts, sparsity, and training videos every 10 episodes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from collections import deque
import random
import time
import psutil
import os
import pickle
import cv2
from dataclasses import dataclass

# ---------------- Output Directories ---------------- #
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)


# ---------------- Configuration ---------------- #
@dataclass
class Config:
    episodes: int = 300
    save_interval: int = 100
    grid_size: int = 5
    learning_rate: float = 0.01
    snn_timesteps: int = 16
    snn_beta: float = 0.95
    record_video: bool = True
    fps: int = 5

    # --- R-STDP ---
    a_plus: float = 0.008
    a_minus: float = 0.01
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    tau_eligibility: float = 100.0
    stdp_lr: float = 5e-4
    rstdp_reward_scale: float = 10.0

    # --- SNN tuning ---
    snn_hidden_size: int = 64
    surrogate_slope: float = 10.0

    # --- Policy gradient tuning ---
    value_lr: float = 1e-3
    entropy_coef: float = 0.01
    grad_clip: float = 2.0


config = Config()


# ---------------- Metrics Tracker ---------------- #
class MetricsTracker:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'inference_times': [],
            'memory_usage': [],
            'sparsity': [],
            'energy': [],
            'episode_lengths': [],
            'success_rate': deque(maxlen=10),
            'parameter_count': 0
        }

    def update(self, episode: int, **kwargs):
        if episode not in self.metrics['episodes']:
            self.metrics['episodes'].append(episode)
        if 'reward' in kwargs:
            self.metrics['rewards'].append(kwargs['reward'])
        for key, value in kwargs.items():
            if key in self.metrics and key not in ['parameter_count', 'reward']:
                if isinstance(self.metrics[key], deque):
                    self.metrics[key].append(value)
                else:
                    self.metrics[key].append(value)

    def get_current_stats(self):
        stats = {
            'avg_reward': np.mean(self.metrics['rewards'][-10:]) if self.metrics['rewards'] else 0,
            'avg_inference_time': np.mean(self.metrics['inference_times'][-10:]) if self.metrics['inference_times'] else 0,
            'avg_memory': np.mean(self.metrics['memory_usage'][-10:]) if self.metrics['memory_usage'] else 0,
            'avg_energy': np.mean(self.metrics['energy'][-10:]) if self.metrics['energy'] else 0,
            'success_rate': np.mean(self.metrics['success_rate']) if self.metrics['success_rate'] else 0,
            'parameter_count': self.metrics['parameter_count']
        }
        if self.model_type == 'SNN' and self.metrics['sparsity']:
            stats['avg_sparsity'] = np.mean(self.metrics['sparsity'][-10:])
        return stats

    # ✅ bring back save_checkpoint
    def save_checkpoint(self, episode: int):
        """Save metrics to file"""
        filename = f"outputs/data/{self.model_type}_metrics_ep{episode}.pkl"
        with open(filename, 'wb') as f:
            save_metrics = self.metrics.copy()
            save_metrics['success_rate'] = list(self.metrics['success_rate'])  # convert deque
            pickle.dump(save_metrics, f)
        print(f"  Saved {self.model_type} metrics to {filename}")


# ---------------- Environment ---------------- #
class GridWorld:
    def __init__(self, size=5, max_steps=50):
        self.size = size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.goal = (self.size - 1, self.size - 1)
        self.obstacles = [(1, 2), (3, 3)]
        self.pos = (0, 0)
        self.trajectory = [self.pos]
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.pos] = 1
        return state.flatten()

    def step(self, action):
        x, y = self.pos
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.size - 1, y + 1)
        new_pos = (x, y)
        old_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        new_dist = abs(new_pos[0] - self.goal[0]) + abs(new_pos[1] - self.goal[1])
        reward = (old_dist - new_dist) * 2.0 - 0.1
        done, success = False, False
        if new_pos in self.obstacles:
            return self._get_state(), -10.0, True, False
        if new_pos == self.goal:
            self.pos = new_pos
            self.trajectory.append(self.pos)
            return self._get_state(), 500.0, True, True
        self.pos = new_pos
        self.trajectory.append(self.pos)
        return self._get_state(), reward, done, success

    def render_frame(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(self.size + 1):
            ax.axhline(i, color="gray", linewidth=0.5)
            ax.axvline(i, color="gray", linewidth=0.5)
        for obs in self.obstacles:
            rect = Rectangle((obs[1], obs[0]), 1, 1, facecolor="gray")
            ax.add_patch(rect)
        goal_circle = Circle(
            (self.goal[1] + 0.5, self.goal[0] + 0.5), 0.3, facecolor="green"
        )
        ax.add_patch(goal_circle)
        agent_circle = Circle(
            (self.pos[1] + 0.5, self.pos[0] + 0.5), 0.25, facecolor="blue"
        )
        ax.add_patch(agent_circle)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        frame = buf[:, :, 1:]
        plt.close()
        return frame


# ---------------- Models ---------------- #
class ANNPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


class SNNPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=config.surrogate_slope)
        h = config.snn_hidden_size
        self.fc1 = nn.Linear(state_dim, h)
        self.lif1 = snn.Leaky(beta=config.snn_beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(h, h)
        self.lif2 = snn.Leaky(beta=config.snn_beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(h, action_dim)
        self.lif3 = snn.Leaky(beta=config.snn_beta, spike_grad=spike_grad)
        self.timesteps = config.snn_timesteps
        self.eligibility = None
        self.a_plus, self.a_minus = config.a_plus, config.a_minus
        self.stdp_lr = config.stdp_lr

    def reset_rstdp_traces(self):
        device = next(self.parameters()).device
        self.eligibility = torch.zeros(
            (self.fc3.out_features, self.fc3.in_features), device=device
        )

    def _accumulate_traces_step(self, spk_pre, spk_post):
        if spk_pre is None or spk_post is None:
            return
        pre = spk_pre.mean(0)
        post = spk_post.mean(0)
        hebb = torch.ger(post, pre)
        self.eligibility = 0.95 * self.eligibility + (self.a_plus - self.a_minus) * hebb

    def apply_rstdp(self, reward):
        if self.eligibility is None:
            return
        dw = self.eligibility * (reward * config.rstdp_reward_scale * self.stdp_lr)
        with torch.no_grad():
            self.fc3.weight.add_(dw.clamp(-1e-2, 1e-2))

    def forward(self, x, track_sparsity=False):
        mem1, mem2, mem3 = (
            self.lif1.init_leaky(),
            self.lif2.init_leaky(),
            self.lif3.init_leaky(),
        )
        spk3_hist, spike_count, total = [], 0, 0
        for _ in range(self.timesteps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            spk3_hist.append(spk3)
            self._accumulate_traces_step(spk2.detach(), spk3.detach())
            if track_sparsity:
                spike_count += spk1.sum().item() + spk2.sum().item() + spk3.sum().item()
                total += spk1.numel() + spk2.numel() + spk3.numel()
        readout = torch.stack(spk3_hist).mean(0)
        sparsity = 1.0 - spike_count / total if track_sparsity and total > 0 else None
        return torch.softmax(readout, dim=-1), sparsity


# ---------------- Video Recorder ---------------- #
class VideoRecorder:
    def __init__(self, filename, fps=5):
        self.filename = filename
        self.fps = fps
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def save(self):
        if not self.frames:
            return
        h, w = self.frames[0].shape[:2]
        out = cv2.VideoWriter(
            self.filename, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        for f in self.frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"  Saved video to {self.filename}")


# ---------------- Training ---------------- #
def train_episode(
    env, model, optimizer, model_type, tracker, episode, record_video=False
):
    state = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    done = False
    total_reward = 0
    log_probs = []
    frames = []
    inf_times = []
    mems = []
    sparsities = []
    steps = 0
    success = False
    process = psutil.Process(os.getpid())
    if model_type == "SNN":
        model.reset_rstdp_traces()
    while not done:
        if record_video:
            frames.append(env.render_frame())
        t0 = time.perf_counter()
        m0 = process.memory_info().rss / 1024 / 1024
        if model_type == "SNN":
            probs, sparsity = model(state_tensor, track_sparsity=True)
            if sparsity is not None:
                sparsities.append(sparsity)
        else:
            probs = model(state_tensor)
        inf_times.append((time.perf_counter() - t0) * 1000)
        mems.append(max(0, process.memory_info().rss / 1024 / 1024 - m0))
        dist = torch.distributions.Categorical(probs)
        epsilon = max(0.05, 0.3 * (0.995**episode))
        action = (
            torch.randint(0, probs.shape[-1], (1,))
            if random.random() < epsilon
            else dist.sample()
        )
        log_probs.append(dist.log_prob(action))
        next_state, reward, done, succ = env.step(action.item())
        success |= succ
        total_reward += reward
        state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        steps += 1
    # ---- Policy update ----
    if log_probs:
        returns = torch.tensor([total_reward] * len(log_probs), dtype=torch.float32)
        logps = torch.stack(log_probs)
        advantages = returns - returns.mean()
        policy_loss = -(logps * advantages).mean()
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
        loss = policy_loss - config.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        loss_value = loss.item()
    else:
        loss_value = 0
    if model_type == "SNN":
        model.apply_rstdp(total_reward)
    avg_inf = np.mean(inf_times) if inf_times else 0
    energy = (
        avg_inf * (1 - np.mean(sparsities)) * 0.1
        if (model_type == "SNN" and sparsities)
        else avg_inf * 0.5
    )
    tracker.update(
        episode,
        reward=total_reward,
        loss=loss_value,
        inference_time=avg_inf,
        memory_usage=np.mean(mems) if mems else 0,
        energy=energy,
        episode_length=steps,
        success_rate=int(success),
    )
    if model_type == "SNN" and sparsities:
        tracker.metrics["sparsity"].append(np.mean(sparsities))
    return total_reward, frames


# ---------------- Plotting Functions ---------------- #
def save_comparison_plots(ann_tracker, snn_tracker, episode):
    """Save comprehensive comparison plots"""
    fig = plt.figure(figsize=(20, 10))

    # print(len(ann_tracker.metrics['rewards']))

    episodes = ann_tracker.metrics['episodes']
    rewards = ann_tracker.metrics['rewards']
    n = min(len(episodes), len(rewards))

    episodes_snn = snn_tracker.metrics['episodes']
    rewards_snn = snn_tracker.metrics['rewards']
    n_snn = min(len(episodes_snn), len(rewards_snn))
    
    # 1. Rewards comparison
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(episodes[:n], rewards[:n], 
             label='ANN', color='blue', alpha=0.7)
    ax1.plot(episodes[:n_snn], rewards[:n_snn], 
             label='SNN', color='red', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Inference Time
    ax2 = plt.subplot(3, 3, 2)
    if ann_tracker.metrics['inference_times'] and snn_tracker.metrics['inference_times']:
        ax2.boxplot([ann_tracker.metrics['inference_times'][-100:], 
                     snn_tracker.metrics['inference_times'][-100:]], 
                    labels=['ANN', 'SNN'])
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Speed')
        ax2.grid(True, alpha=0.3)
    
    # 3. Memory Usage
    ax3 = plt.subplot(3, 3, 3)
    if ann_tracker.metrics['memory_usage'] and snn_tracker.metrics['memory_usage']:
        ann_mem = np.mean(ann_tracker.metrics['memory_usage'][-100:])
        snn_mem = np.mean(snn_tracker.metrics['memory_usage'][-100:])
        bars = ax3.bar(['ANN', 'SNN'], [ann_mem, snn_mem], color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('Average Memory Usage')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
    
    # 4. Energy Consumption
    ax4 = plt.subplot(3, 3, 4)
    if ann_tracker.metrics['energy'] and snn_tracker.metrics['energy']:
        ann_energy = np.mean(ann_tracker.metrics['energy'][-100:])
        snn_energy = np.mean(snn_tracker.metrics['energy'][-100:])
        bars = ax4.bar(['ANN', 'SNN'], [ann_energy, snn_energy], 
                      color=['blue', 'red'], alpha=0.7)
        ax4.set_ylabel('Relative Energy')
        ax4.set_title('Energy Consumption')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
    
    # 5. SNN Sparsity
    ax5 = plt.subplot(3, 3, 5)
    if snn_tracker.metrics['sparsity']:
        ax5.plot(snn_tracker.metrics['episodes'], 
                [s*100 for s in snn_tracker.metrics['sparsity']], 
                color='purple', alpha=0.7)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Sparsity (%)')
        ax5.set_title('SNN Neural Sparsity')
        ax5.grid(True, alpha=0.3)
        avg_sparsity = np.mean(snn_tracker.metrics['sparsity'][-100:]) * 100
        ax5.axhline(y=avg_sparsity, color='purple', linestyle='--', alpha=0.5)
        ax5.text(len(snn_tracker.metrics['episodes'])*0.7, avg_sparsity+1,
                f'Avg: {avg_sparsity:.1f}%', fontsize=10)
    
    # 6. Episode Lengths
    # ax6 = plt.subplot(3, 3, 6)
    # ax6.plot(ann_tracker.metrics['episodes'], ann_tracker.metrics['episode_lengths'], 
    #          label='ANN', color='blue', alpha=0.7)
    # ax6.plot(snn_tracker.metrics['episodes'], snn_tracker.metrics['episode_lengths'], 
    #          label='SNN', color='red', alpha=0.7)
    # ax6.set_xlabel('Episode')
    # ax6.set_ylabel('Steps')
    # ax6.set_title('Episode Length (Lower is Better)')
    # ax6.legend()
    # ax6.grid(True, alpha=0.3)
    
    # 7. Success Rate (Rolling)
    ax7 = plt.subplot(3, 3, 7)
    if ann_tracker.metrics['success_rate'] and snn_tracker.metrics['success_rate']:
        # Calculate rolling success rate
        window = 10
        ann_success = [np.mean(ann_tracker.metrics['success_rate']) * 100]
        snn_success = [np.mean(snn_tracker.metrics['success_rate']) * 100]
        
        ax7.bar(['ANN', 'SNN'], [ann_success[0], snn_success[0]], 
               color=['blue', 'red'], alpha=0.7)
        ax7.set_ylabel('Success Rate (%)')
        ax7.set_title(f'Success Rate (Last {window} Episodes)')
        ax7.set_ylim([0, 105])
        for i, v in enumerate([ann_success[0], snn_success[0]]):
            ax7.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 8. Parameter Count
    ax8 = plt.subplot(3, 3, 8)
    ann_params = ann_tracker.metrics['parameter_count']
    snn_params = snn_tracker.metrics['parameter_count']
    bars = ax8.bar(['ANN', 'SNN'], [ann_params, snn_params], 
                  color=['blue', 'red'], alpha=0.7)
    ax8.set_ylabel('Parameters')
    ax8.set_title('Model Complexity')
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom')
    
    # 9. Efficiency Summary
    ax9 = plt.subplot(3, 3, 9)
    ann_stats = ann_tracker.get_current_stats()
    snn_stats = snn_tracker.get_current_stats()
    
    # Calculate efficiency ratios
    efficiency_metrics = {
        'Speed': snn_stats['avg_inference_time'] / (ann_stats['avg_inference_time'] + 1e-6),
        'Memory': snn_stats['avg_memory'] / (ann_stats['avg_memory'] + 1e-6),
        'Energy': snn_stats['avg_energy'] / (ann_stats['avg_energy'] + 1e-6),
        'Reward': snn_stats['avg_reward'] / (ann_stats['avg_reward'] + 1e-6)
    }
    
    x_pos = np.arange(len(efficiency_metrics))
    values = list(efficiency_metrics.values())
    bars = ax9.bar(x_pos, values, color=['green' if v < 1 else 'orange' for v in values])
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(efficiency_metrics.keys())
    ax9.set_ylabel('SNN / ANN Ratio')
    ax9.set_title('Efficiency Comparison (Lower is Better for Speed/Memory/Energy)')
    ax9.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    for bar, val in zip(bars, values):
        ax9.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top')
    
    plt.suptitle(f'GridWorld Navigation Metrics - Episode {episode}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"outputs/plots/comparison_ep{episode}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot to {filename}")

def save_summary_stats(ann_tracker, snn_tracker, episode):
    """Save summary statistics to text file"""
    filename = f"outputs/data/summary_ep{episode}.txt"
    
    ann_stats = ann_tracker.get_current_stats()
    snn_stats = snn_tracker.get_current_stats()
    
    with open(filename, 'w') as f:
        f.write(f"Summary Statistics - Episode {episode}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ANN Performance:\n")
        f.write(f"  Average Reward: {ann_stats['avg_reward']:.2f}\n")
        f.write(f"  Success Rate: {ann_stats['success_rate']:.1%}\n")
        f.write(f"  Inference Time: {ann_stats['avg_inference_time']:.3f} ms\n")
        f.write(f"  Memory Usage: {ann_stats['avg_memory']:.2f} MB\n")
        f.write(f"  Energy: {ann_stats['avg_energy']:.4f}\n")
        f.write(f"  Parameters: {ann_stats['parameter_count']:,}\n\n")
        
        f.write("SNN Performance:\n")
        f.write(f"  Average Reward: {snn_stats['avg_reward']:.2f}\n")
        f.write(f"  Success Rate: {snn_stats['success_rate']:.1%}\n")
        f.write(f"  Inference Time: {snn_stats['avg_inference_time']:.3f} ms\n")
        f.write(f"  Memory Usage: {snn_stats['avg_memory']:.2f} MB\n")
        f.write(f"  Energy: {snn_stats['avg_energy']:.4f}\n")
        f.write(f"  Parameters: {snn_stats['parameter_count']:,}\n")
        if 'avg_sparsity' in snn_stats:
            f.write(f"  Sparsity: {snn_stats['avg_sparsity']:.1%}\n\n")
        
        f.write("Efficiency Gains (SNN vs ANN):\n")
        f.write(f"  Speed: {ann_stats['avg_inference_time']/(snn_stats['avg_inference_time']+1e-6):.2f}x\n")
        f.write(f"  Memory: {ann_stats['avg_memory']/(snn_stats['avg_memory']+1e-6):.2f}x\n")
        f.write(f"  Energy: {ann_stats['avg_energy']/(snn_stats['avg_energy']+1e-6):.2f}x\n")
    
    print(f"  Saved summary to {filename}")

# ---------------- Main Training Loop ---------------- #
def run_comprehensive_training():
    """Run training with comprehensive tracking and saving"""
    print("=" * 60)
    print("GridWorld Navigation: Comprehensive ANN vs SNN Comparison")
    print("=" * 60)
    
    # Initialize environments
    env_ann = GridWorld(size=config.grid_size)
    env_snn = GridWorld(size=config.grid_size)
    
    state_dim = config.grid_size * config.grid_size
    action_dim = 4
    
    # Initialize models
    ann_model = ANNPolicy(state_dim, action_dim)
    snn_model = SNNPolicy(state_dim, action_dim)
    
    # Count parameters
    ann_params = sum(p.numel() for p in ann_model.parameters())
    snn_params = sum(p.numel() for p in snn_model.parameters())
    
    print(f"\nModel Parameters:")
    print(f"  ANN: {ann_params:,}")
    print(f"  SNN: {snn_params:,}")
    
    # Initialize optimizers
    ann_optimizer = optim.Adam(ann_model.parameters(), lr=config.learning_rate)
    snn_optimizer = optim.Adam(snn_model.parameters(), lr=config.learning_rate)
    
    # Initialize trackers
    ann_tracker = MetricsTracker('ANN')
    snn_tracker = MetricsTracker('SNN')
    
    ann_tracker.metrics['parameter_count'] = ann_params
    snn_tracker.metrics['parameter_count'] = snn_params
    
    print(f"\nTraining for {config.episodes} episodes...")
    print(f"Saving every {config.save_interval} episodes or when goal is reached\n")
    
    # Training loop
    for episode in range(config.episodes):
        # Train ANN
        ann_reward, ann_frames = train_episode(
            env_ann, ann_model, ann_optimizer, 'ANN', 
            ann_tracker, episode, record_video=True
        )
        
        # Train SNN  
        snn_reward, snn_frames = train_episode(
            env_snn, snn_model, snn_optimizer, 'SNN',
            snn_tracker, episode, record_video=True
        )
        
        # Print progress
        if (episode + 1) % 10 == 0:
            ann_stats = ann_tracker.get_current_stats()
            snn_stats = snn_tracker.get_current_stats()
            
            print(f"Episode {episode + 1}/{config.episodes}")
            print(f"  ANN - Reward: {ann_reward:.2f}, Avg: {ann_stats['avg_reward']:.2f}, "
                  f"Success: {ann_stats['success_rate']:.1%}")
            print(f"  SNN - Reward: {snn_reward:.2f}, Avg: {snn_stats['avg_reward']:.2f}, "
                  f"Success: {snn_stats['success_rate']:.1%}, "
                  f"Sparsity: {snn_stats.get('avg_sparsity', 0):.1%}")
        
        # --- Save if interval OR if goal reached ---
        save_now = False
        save_reason = []
        if (episode + 1) % config.save_interval == 0:
            save_now, save_reason = True, ["interval"]
        if ann_tracker.metrics['success_rate'] and ann_tracker.metrics['success_rate'][-1] == 1:
            save_now, save_reason = True, save_reason + ["ANN success"]
        if snn_tracker.metrics['success_rate'] and snn_tracker.metrics['success_rate'][-1] == 1:
            save_now, save_reason = True, save_reason + ["SNN success"]

        if save_now:
            print(f"\nSaving checkpoint at episode {episode + 1} ({', '.join(save_reason)})...")

            # Save ANN video if success
            if ann_frames and ann_tracker.metrics['success_rate'][-1] == 1:
                video_filename = f"outputs/videos/ann_goal_ep{episode+1}.mp4"
                video_recorder = VideoRecorder(video_filename, config.fps)
                for frame in ann_frames:
                    video_recorder.add_frame(frame)
                video_recorder.save()

            # Save SNN video if success
            if snn_frames and snn_tracker.metrics['success_rate'][-1] == 1:
                video_filename = f"outputs/videos/snn_goal_ep{episode+1}.mp4"
                video_recorder = VideoRecorder(video_filename, config.fps)
                for frame in snn_frames:
                    video_recorder.add_frame(frame)
                video_recorder.save()
            
            # Save metrics
            ann_tracker.save_checkpoint(episode + 1)
            snn_tracker.save_checkpoint(episode + 1)
            
            # Save plots
            save_comparison_plots(ann_tracker, snn_tracker, episode + 1)
            
            # Save summary statistics
            save_summary_stats(ann_tracker, snn_tracker, episode + 1)
            
            # Save model checkpoints
            torch.save({
                'episode': episode + 1,
                'model_state_dict': ann_model.state_dict(),
                'optimizer_state_dict': ann_optimizer.state_dict(),
            }, f"outputs/checkpoints/ann_model_ep{episode+1}.pth")
            
            torch.save({
                'episode': episode + 1,
                'model_state_dict': snn_model.state_dict(),
                'optimizer_state_dict': snn_optimizer.state_dict(),
            }, f"outputs/checkpoints/snn_model_ep{episode+1}.pth")
            
            print(f"  All data saved for episode {episode + 1}\n")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    ann_stats = ann_tracker.get_current_stats()
    snn_stats = snn_tracker.get_current_stats()
    
    print("\nFinal Performance Summary:")
    print(f"  ANN - Avg Reward: {ann_stats['avg_reward']:.2f}, Success: {ann_stats['success_rate']:.1%}")
    print(f"  SNN - Avg Reward: {snn_stats['avg_reward']:.2f}, Success: {snn_stats['success_rate']:.1%}")
    
    print("\nEfficiency Comparison:")
    print(f"  Speed Gain: {ann_stats['avg_inference_time']/(snn_stats['avg_inference_time']+1e-6):.2f}x")
    print(f"  Memory Gain: {ann_stats['avg_memory']/(snn_stats['avg_memory']+1e-6):.2f}x")
    print(f"  Energy Gain: {ann_stats['avg_energy']/(snn_stats['avg_energy']+1e-6):.2f}x")
    if 'avg_sparsity' in snn_stats:
        print(f"  SNN Sparsity: {snn_stats['avg_sparsity']:.1%}")
    
    print("\nOutputs saved in:")
    print("  - outputs/plots/     : Comparison plots")
    print("  - outputs/videos/    : Training videos")
    print("  - outputs/data/      : Metrics and summaries")
    print("  - outputs/checkpoints/: Model checkpoints")


# ---------------- Run the Training ---------------- #
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run comprehensive training
    run_comprehensive_training()
