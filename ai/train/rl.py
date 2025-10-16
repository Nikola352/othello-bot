import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable

from train.process_games import preload_expert_memory
from model.network import PolicyNetwork, ValueNetwork
from model.agent import ActorCriticAgent
from model.settings import EPS_DECAY, LR, START_EPS, END_EPS, EPISODES, TARGET_LIFESPAN
from environment.environment import EnvState


class RLTrainingMetrics:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.episodes = []
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
        self.epsilon_values = []
        self.memory_sizes = []
        
        # Evaluation metrics
        self.eval_episodes = []
        self.eval_results = {}  # {strategy_name: [win_rates]}
        
        # Smoothing for visualization
        self.smooth_window = 100
        
    def add_step(self, episode: int, actor_loss: Optional[float], critic_loss: Optional[float], 
                 epsilon: float, memory_size: int):
        """Add training step metrics"""
        self.episodes.append(episode)
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        self.epsilon_values.append(epsilon)
        self.memory_sizes.append(memory_size)
    
    def add_episode_reward(self, reward: float):
        """Add episode reward"""
        self.episode_rewards.append(reward)
    
    def add_evaluation(self, episode: int, strategy_name: str, win_rate: float):
        """Add evaluation result"""
        self.eval_episodes.append(episode)
        if strategy_name not in self.eval_results:
            self.eval_results[strategy_name] = []
        self.eval_results[strategy_name].append(win_rate)
    
    def _smooth(self, values: List[float], window: int = None) -> List[float]:
        """Apply moving average smoothing"""
        if window is None:
            window = self.smooth_window
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode='valid').tolist()
    
    def save_metrics(self):
        """Save all metrics to JSON"""
        metrics_dict = {
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'episode_rewards': self.episode_rewards,
            'epsilon_values': self.epsilon_values,
            'memory_sizes': self.memory_sizes,
            'eval_episodes': self.eval_episodes,
            'eval_results': self.eval_results,
        }
        
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    
    def plot_training_metrics(self, max_episodes: Optional[int] = None):
        """Generate comprehensive training metric plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RL Training Metrics', fontsize=16, fontweight='bold')
        
        # Determine x-axis range
        x_max = max_episodes if max_episodes else len(self.episodes)
        
        # Plot 1: Actor Loss
        if self.actor_losses:
            episodes_actor = list(range(len(self.actor_losses)))
            smooth_actor = self._smooth(self.actor_losses)
            axes[0, 0].plot(episodes_actor, self.actor_losses, 'b-', alpha=0.3, label='Raw')
            axes[0, 0].plot(range(len(smooth_actor)), smooth_actor, 'b-', linewidth=2, label='Smoothed')
            axes[0, 0].set_title('Actor Loss')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Critic Loss
        if self.critic_losses:
            episodes_critic = list(range(len(self.critic_losses)))
            smooth_critic = self._smooth(self.critic_losses)
            axes[0, 1].plot(episodes_critic, self.critic_losses, 'r-', alpha=0.3, label='Raw')
            axes[0, 1].plot(range(len(smooth_critic)), smooth_critic, 'r-', linewidth=2, label='Smoothed')
            axes[0, 1].set_title('Critic Loss')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Epsilon Decay
        if self.epsilon_values:
            # Sample every 100 episodes to avoid clutter
            sample_idx = list(range(0, len(self.epsilon_values), max(1, len(self.epsilon_values) // 1000)))
            sample_eps = [self.epsilon_values[i] for i in sample_idx]
            sample_episodes = [self.episodes[i] if i < len(self.episodes) else i for i in sample_idx]
            
            axes[0, 2].plot(sample_episodes, sample_eps, 'g-', linewidth=2)
            axes[0, 2].set_title('Epsilon Decay Schedule')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Epsilon')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Episode Rewards
        if self.episode_rewards:
            episode_nums = list(range(1, len(self.episode_rewards) + 1))
            smooth_rewards = self._smooth(self.episode_rewards, window=min(100, len(self.episode_rewards) // 10))
            axes[1, 0].plot(episode_nums, self.episode_rewards, 'orange', alpha=0.3, label='Raw')
            axes[1, 0].plot(range(len(smooth_rewards)), smooth_rewards, 'orange', linewidth=2, label='Smoothed')
            axes[1, 0].set_title('Episode Rewards')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Total Reward')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Memory Size
        if self.memory_sizes:
            sample_idx = list(range(0, len(self.memory_sizes), max(1, len(self.memory_sizes) // 1000)))
            sample_mem = [self.memory_sizes[i] for i in sample_idx]
            sample_episodes = [self.episodes[i] if i < len(self.episodes) else i for i in sample_idx]
            
            axes[1, 1].plot(sample_episodes, sample_mem, 'purple', linewidth=2)
            axes[1, 1].set_title('Replay Memory Size')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Memory Size')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Evaluation Results
        if self.eval_results:
            for strategy_name, win_rates in self.eval_results.items():
                axes[1, 2].plot(self.eval_episodes[:len(win_rates)], win_rates, 
                               marker='o', label=strategy_name, linewidth=2, markersize=6)
            axes[1, 2].set_title('Evaluation: Win Rate vs Strategies')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Win Rate (%)')
            axes[1, 2].set_ylim([0, 105])
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No evaluation data', ha='center', va='center', 
                           transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_individual_metrics(self):
        """Save individual high-resolution plots"""
        
        # Actor Loss
        if self.actor_losses:
            plt.figure(figsize=(12, 6))
            smooth_actor = self._smooth(self.actor_losses)
            plt.plot(range(len(self.actor_losses)), self.actor_losses, 'b-', 
                    alpha=0.3, label='Raw', linewidth=1)
            plt.plot(range(len(smooth_actor)), smooth_actor, 'b-', 
                    label='Smoothed (100-step MA)', linewidth=2)
            plt.title('Actor Loss Over Training', fontsize=14, fontweight='bold')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'actor_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Critic Loss
        if self.critic_losses:
            plt.figure(figsize=(12, 6))
            smooth_critic = self._smooth(self.critic_losses)
            plt.plot(range(len(self.critic_losses)), self.critic_losses, 'r-', 
                    alpha=0.3, label='Raw', linewidth=1)
            plt.plot(range(len(smooth_critic)), smooth_critic, 'r-', 
                    label='Smoothed (100-step MA)', linewidth=2)
            plt.title('Critic Loss Over Training', fontsize=14, fontweight='bold')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'critic_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Episode Rewards
        if self.episode_rewards:
            plt.figure(figsize=(12, 6))
            episode_nums = list(range(1, len(self.episode_rewards) + 1))
            smooth_window = min(100, len(self.episode_rewards) // 10)
            smooth_rewards = self._smooth(self.episode_rewards, window=smooth_window)
            plt.plot(episode_nums, self.episode_rewards, 'orange', 
                    alpha=0.3, label='Raw', linewidth=1)
            plt.plot(range(len(smooth_rewards)), smooth_rewards, 'orange', 
                    label=f'Smoothed ({smooth_window}-episode MA)', linewidth=2)
            plt.title('Episode Rewards Over Training', fontsize=14, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'episode_rewards.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Evaluation Results
        if self.eval_results:
            plt.figure(figsize=(12, 6))
            for strategy_name, win_rates in self.eval_results.items():
                plt.plot(self.eval_episodes[:len(win_rates)], win_rates, 
                        marker='o', label=strategy_name, linewidth=2.5, markersize=8)
            plt.title('Agent Performance vs Strategies', fontsize=14, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
            plt.ylim([0, 105])
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.close()


class RLEvaluator: 
    def __init__(self, agent: ActorCriticAgent, num_games: int = 50, device=None):
        self.agent = agent
        self.num_games = num_games
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_vs_strategy(self, strategy_func: Callable, strategy_name: str) -> Dict:
        agent_wins = 0
        strategy_wins = 0
        draws = 0
        
        print(f"\n  Evaluating vs {strategy_name}... (0/{self.num_games})", end='', flush=True)
        
        for game_idx in range(self.num_games):
            # Alternate who goes first
            agent_is_first = (game_idx % 2 == 0)
            
            state = EnvState()
            game_over = False
            
            while not game_over:
                is_agent_turn = (state.turn == 1) if agent_is_first else (state.turn == -1)
                
                if is_agent_turn:
                    # Agent move (deterministic, no exploration)
                    action = self.agent.select_optimal_action(state)
                    if action is None:
                        state.skip_move()
                    else:
                        state.act(action)
                else:
                    # Strategy move
                    action = strategy_func(state)
                    if action is None:
                        state.skip_move()
                    else:
                        state.act(action)
                
                game_over = state.is_final()
            
            # Determine winner
            final_score = state.get_score()
            agent_final = final_score if agent_is_first else -final_score
            
            if agent_final > 0:
                agent_wins += 1
            elif agent_final < 0:
                strategy_wins += 1
            else:
                draws += 1
            
            if (game_idx + 1) % 10 == 0:
                print(f"\r  Evaluating vs {strategy_name}... ({game_idx + 1}/{self.num_games})", 
                      end='', flush=True)
        
        print(f"\r  Evaluating vs {strategy_name}... ({self.num_games}/{self.num_games}) ✓")
        
        win_rate = 100.0 * agent_wins / self.num_games
        draw_rate = 100.0 * draws / self.num_games
        
        results = {
            'strategy': strategy_name,
            'agent_wins': agent_wins,
            'strategy_wins': strategy_wins,
            'draws': draws,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'games_played': self.num_games,
        }
        
        print(f"    Results: {agent_wins}W - {strategy_wins}L - {draws}D | Win Rate: {win_rate:.1f}%")
        
        return results


def train_rl(
        save_path: str,
        start_checkpoint_path: str = None,
        start_episode: int = 1,
        checkpoint_dir: str = None,
        policy_net: PolicyNetwork = None,
        value_net: ValueNetwork = None,
        expert_data_path: str = None,
        start_from_policy: bool = False,
        eval_strategies: Optional[Dict[str, Callable]] = None,
        eval_interval: int = 2500,
        num_eval_games: int = 50,
    ) -> Dict:
    """
    Train RL agent with comprehensive metrics tracking and evaluation.
    
    Args:
        save_path: Path to save final policy network
        start_checkpoint_path: Path to load checkpoint from
        start_episode: Episode to start from
        checkpoint_dir: Directory to save checkpoints
        policy_net: Pre-initialized policy network
        value_net: Pre-initialized value network
        expert_data_path: Path to expert data for imitation pretraining
        start_from_policy: Load only policy network from checkpoint
        eval_strategies: Dict mapping strategy names to strategy functions
                        Each function should take (state) -> action or None
        eval_interval: Evaluate every N episodes (default 5000)
        num_eval_games: Number of games per evaluation (default 50)
    
    Returns:
        Dictionary with trained agent, metrics, and evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(checkpoint_dir or save_path).parent
    run_dir = output_dir / f"rl_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize agent and metrics
    agent = ActorCriticAgent(device, policy_net=policy_net, value_net=value_net, actor_frozen=False)
    metrics = RLTrainingMetrics(run_dir)
    
    if start_checkpoint_path and os.path.exists(start_checkpoint_path):
        print(f"Loading checkpoint from: {start_checkpoint_path}")
        if start_from_policy:
            agent.load_policy_net(start_checkpoint_path)
        else:
            agent.load_model(start_checkpoint_path)
    
    if expert_data_path:
        print(f"Preloading expert data from: {expert_data_path}\n")
        preload_expert_memory(agent, expert_data_path)
    
    # Initialize evaluator if strategies provided
    evaluator = None
    if eval_strategies:
        evaluator = RLEvaluator(agent, num_games=num_eval_games, device=device)
        print(f"Evaluation enabled: {list(eval_strategies.keys())}")
        print(f"Evaluation interval: every {eval_interval} episodes with {num_eval_games} games\n")
    
    # Training loop
    episode_reward_buffer = deque(maxlen=100)
    
    print(f"Starting RL training for {EPISODES} episodes...")
    print("=" * 80)
    
    for episode in range(start_episode, EPISODES + 1):
        eps = END_EPS + (START_EPS - END_EPS) * np.exp(-1.0 * episode / EPS_DECAY)
        total_reward = 0.0
        
        state = EnvState()
        while not state.is_final():
            action = agent.select_action(state, eps)
            if action is None:
                state.skip_move()
                continue
            
            prev_state = state.copy()
            reward = state.act(action)
            total_reward += reward
            
            agent.add_to_memory(prev_state, action, reward, state)
            actor_loss, critic_loss = agent.optimize()
            
            metrics.add_step(
                episode, 
                actor_loss.detach().item() if actor_loss is not None else None,
                critic_loss.detach().item() if critic_loss is not None else None,
                eps,
                len(agent.memory)
            )
        
        metrics.add_episode_reward(total_reward)
        episode_reward_buffer.append(total_reward)
        
        # Update target network
        if episode % TARGET_LIFESPAN == 0:
            agent.update_target()
        
        # Logging
        if episode % 1000 == 0:
            avg_reward = np.mean(episode_reward_buffer)
            avg_critic_loss = np.mean(metrics.critic_losses[-1000:]) if metrics.critic_losses else 0
            avg_actor_loss = np.mean(metrics.actor_losses[-1000:]) if metrics.actor_losses else 0
            
            print(f"EP {episode:6d}/{EPISODES} | eps={eps:.4f} | "
                  f"avg_reward={avg_reward:7.2f} | "
                  f"actor_loss={avg_actor_loss:.4f} | critic_loss={avg_critic_loss:.4f} | "
                  f"mem={len(agent.memory)}")
        
        # Periodic checkpointing
        if episode % 10000 == 0 and checkpoint_dir:
            agent.save_model(checkpoint_dir / f"checkpoint_{episode:06d}.pth")
        
        # Periodic evaluation
        if evaluator and episode % eval_interval == 0:
            print(f"\n--- Evaluation at Episode {episode} ---")
            for strategy_name, strategy_func in eval_strategies.items():
                results = evaluator.evaluate_vs_strategy(strategy_func, strategy_name)
                metrics.add_evaluation(episode, strategy_name, results['win_rate'])
                
                # Save evaluation results
                eval_log_path = run_dir / f"eval_episode_{episode}.json"
                with open(eval_log_path, 'a') as f:
                    json.dump(results, f)
                    f.write('\n')
            print()
    
    print("=" * 80)
    print("Training completed!\n")
    
    # Final checkpoint
    if checkpoint_dir:
        agent.save_model(checkpoint_dir / "checkpoint_final.pth")
    
    agent.save_policy_net(save_path)
    print(f"Final policy saved to: {save_path}\n")
    
    # Save metrics and generate visualizations
    print("Generating visualizations...")
    metrics.save_metrics()
    metrics.plot_training_metrics()
    metrics.plot_individual_metrics()
    
    # Create summary report
    summary = {
        'total_episodes': EPISODES,
        'final_epsilon': eps,
        'total_steps': len(metrics.actor_losses) + len(metrics.critic_losses),
        'final_memory_size': len(agent.memory),
        'evaluation_intervals': eval_interval,
        'evaluation_results': metrics.eval_results,
    }
    
    with open(run_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results saved to: {run_dir}")
    print(f"  - Metrics: {run_dir / 'metrics.json'}")
    print(f"  - Plots: {run_dir / 'training_metrics.png'}")
    print(f"  - Summary: {run_dir / 'training_summary.json'}")
    if evaluator:
        print(f"  - Evaluations: {run_dir / 'eval_episode_*.json'}")
    
    return {
        'agent': agent,
        'metrics': metrics,
        'run_dir': run_dir,
        'eval_results': metrics.eval_results if evaluator else None,
    }
