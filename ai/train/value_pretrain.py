import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from datetime import datetime
from model.network import ValueNetwork
from model.settings import BATCH_SIZE, LR
from model.settings import VAL_NET_EPOCHS as EPOCHS


class ValueDataset(Dataset):
    def __init__(self, states, values):
        self.states = states
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, value


class RegressionMetrics:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.train_mae = []
        self.val_mae = []
        self.train_r2 = []
        self.val_r2 = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_val_r2 = float('-inf')
        self.best_epoch = 0
        
    def update(self, epoch: int, train_loss: float, val_loss: float,
               train_mse: float, val_mse: float,
               train_mae: float, val_mae: float,
               train_r2: float, val_r2: float, lr: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_mse.append(train_mse)
        self.val_mse.append(val_mse)
        self.train_mae.append(train_mae)
        self.val_mae.append(val_mae)
        self.train_r2.append(train_r2)
        self.val_r2.append(val_r2)
        self.learning_rates.append(lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_r2 = val_r2
            self.best_epoch = epoch
    
    def save_metrics(self):
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2,
            'best_epoch': self.best_epoch
        }
        
        with open(self.save_dir / 'regression_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    
    def plot_metrics(self):
        """Generate and save all training plots"""
        epochs = range(1, len(self.train_losses) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Value Network Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                          label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[0, 0].set_title('MSE Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MAE curves
        axes[0, 1].plot(epochs, self.train_mae, 'b-', label='Training MAE', linewidth=2)
        axes[0, 1].plot(epochs, self.val_mae, 'r-', label='Validation MAE', linewidth=2)
        axes[0, 1].axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                          label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[0, 1].set_title('Mean Absolute Error Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: R² curves
        axes[0, 2].plot(epochs, self.train_r2, 'b-', label='Training R²', linewidth=2)
        axes[0, 2].plot(epochs, self.val_r2, 'r-', label='Validation R²', linewidth=2)
        axes[0, 2].axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                          label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[0, 2].set_title('R² Score Over Epochs')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('R² Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim([-0.1, 1.05])
        
        # Plot 4: Learning rate schedule
        axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot 5: MSE comparison
        axes[1, 1].plot(epochs, self.train_mse, 'b-', label='Training MSE', linewidth=2)
        axes[1, 1].plot(epochs, self.val_mse, 'r-', label='Validation MSE', linewidth=2)
        axes[1, 1].axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                          label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[1, 1].set_title('MSE Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Overfitting analysis (loss gap)
        loss_gap = [t - v for t, v in zip(self.train_losses, self.val_losses)]
        r2_gap = [v - t for t, v in zip(self.train_r2, self.val_r2)]
        
        ax6 = axes[1, 2]
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(epochs, loss_gap, 'b-', label='Loss Gap (Train - Val)', linewidth=2)
        line2 = ax6_twin.plot(epochs, r2_gap, 'r-', label='R² Gap (Val - Train)', linewidth=2)
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss Gap (MSE)', color='b')
        ax6_twin.set_ylabel('R² Gap', color='r')
        ax6.tick_params(axis='y', labelcolor='b')
        ax6_twin.tick_params(axis='y', labelcolor='r')
        ax6.set_title('Overfitting Analysis')
        ax6.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'regression_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual plots
        self._save_individual_plots(epochs)
    
    def _save_individual_plots(self, epochs):
        """Save individual metric plots"""
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                   label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        plt.title('MSE Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # MAE plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_mae, 'b-', label='Training MAE', linewidth=2)
        plt.plot(epochs, self.val_mae, 'r-', label='Validation MAE', linewidth=2)
        plt.axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                   label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        plt.title('Mean Absolute Error Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / 'mae_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # R² plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_r2, 'b-', label='Training R²', linewidth=2)
        plt.plot(epochs, self.val_r2, 'r-', label='Validation R²', linewidth=2)
        plt.axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                   label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        plt.title('R² Score Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([-0.1, 1.05])
        plt.savefig(self.save_dir / 'r2_curve.png', dpi=300, bbox_inches='tight')
        plt.close()


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return float(mse), float(mae), float(r2)


def evaluate_value_model(network: nn.Module, loader: DataLoader,
                         criterion: nn.Module, device: torch.device) -> Tuple[float, float, float, float]:
    network.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for state, value in loader:
            state = state.to(device)
            value = value.to(device)
            
            predictions = network(state)
            loss = criterion(predictions.squeeze(), value)
            
            mse, mae, r2 = compute_regression_metrics(predictions.squeeze(), value)
            
            all_predictions.extend(predictions.squeeze().cpu().numpy().flatten())
            all_targets.extend(value.cpu().numpy().flatten())
            
            total_loss += loss.item()
            total_mse += mse
            total_mae += mae
            total_r2 += r2
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_r2 = total_r2 / num_batches
    
    return avg_loss, avg_mse, avg_mae, avg_r2


def save_value_checkpoint(network: nn.Module, optimizer: optim.Optimizer,
                          epoch: int, metrics: RegressionMetrics,
                          checkpoint_path: Path, is_best: bool = False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': metrics.train_losses[-1],
        'val_loss': metrics.val_losses[-1],
        'train_r2': metrics.train_r2[-1],
        'val_r2': metrics.val_r2[-1],
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / 'best_value_model.pt'
        torch.save(checkpoint, best_path)
        torch.save(network.state_dict(), checkpoint_path.parent / 'best_value_model_state_dict.pt')


def pretrain_value_network(states, values,
                          save_path: Optional[str] = None,
                          gamma: float = 0.99,
                          train_split: float = 0.7,
                          val_split: float = 0.15,
                          test_split: float = 0.15,
                          checkpoint_interval: int = 5,
                          early_stopping_patience: int = 10) -> Dict:
    """
    Pretrain agent's value network using (state -> discounted_value) pairs.
    
    Args:
        states: Input states
        df: DataFrame with columns ['eOthello_game_id', 'winner', 'game_moves']
        save_path: Path to save final model
        gamma: Discount factor for Bellman equation
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        checkpoint_interval: Save checkpoint every N epochs
        early_stopping_patience: Stop if no improvement for N epochs
    
    Returns:
        Dictionary containing trained network and metrics
    """
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Train, val, and test splits must sum to 1.0"
    
    values = np.array(values, dtype=np.float32)
    
    print(f"Value statistics:")
    print(f"  Min:  {values.min():.4f}")
    print(f"  Max:  {values.max():.4f}")
    print(f"  Mean: {values.mean():.4f}")
    print(f"  Std:  {values.std():.4f}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(save_path) if save_path else Path("./value_training_output")
    run_dir = output_dir / f"value_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create and split dataset
    full_dataset = ValueDataset(states, values)
    total_size = len(full_dataset)
    
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset splits:")
    print(f"  Training:   {train_size} samples ({train_split*100:.1f}%)")
    print(f"  Validation: {val_size} samples ({val_split*100:.1f}%)")
    print(f"  Test:       {test_size} samples ({test_split*100:.1f}%)")
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize network
    network = ValueNetwork().to(device)
    network.train()
    
    # Setup training
    optimizer = optim.Adam(network.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    metrics = RegressionMetrics(run_dir)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting value network training for {EPOCHS} epochs...")
    print("=" * 85)
    
    for epoch in range(EPOCHS):
        # Training phase
        network.train()
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        train_r2 = 0.0
        num_train_batches = 0
        
        for state, value in train_loader:
            state = state.to(device)
            value = value.to(device)
            
            optimizer.zero_grad()
            predictions = network(state).squeeze()
            loss = criterion(predictions, value)
            loss.backward()
            optimizer.step()
            
            mse, mae, r2 = compute_regression_metrics(predictions, value)
            
            train_loss += loss.item()
            train_mse += mse
            train_mae += mae
            train_r2 += r2
            num_train_batches += 1
        
        avg_train_loss = train_loss / num_train_batches
        avg_train_mse = train_mse / num_train_batches
        avg_train_mae = train_mae / num_train_batches
        avg_train_r2 = train_r2 / num_train_batches
        
        # Validation phase
        val_loss, val_mse, val_mae, val_r2 = evaluate_value_model(network, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics.update(epoch, avg_train_loss, val_loss,
                      avg_train_mse, val_mse,
                      avg_train_mae, val_mae,
                      avg_train_r2, val_r2, current_lr)
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train R²: {avg_train_r2:.4f} | Val R²: {val_r2:.4f} | "
              f"LR: {current_lr:.2e}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ↓ New best model! (Val Loss: {val_loss:.4f}, R²: {val_r2:.4f})")
        else:
            patience_counter += 1
        
        if (epoch + 1) % checkpoint_interval == 0 or is_best:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_value_checkpoint(network, optimizer, epoch, metrics, checkpoint_path, is_best)
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {early_stopping_patience} consecutive epochs")
            break
    
    print("=" * 85)
    print("\nTraining completed!")
    
    # Final test evaluation
    best_model_path = checkpoint_dir / 'best_value_model.pt'
    if best_model_path.exists():
        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on test set...")
    test_loss, test_mse, test_mae, test_r2 = evaluate_value_model(network, test_loader, criterion, device)
    
    print(f"\nFinal Results:")
    print(f"  Best Validation Loss:  {metrics.best_val_loss:.4f} (Epoch {metrics.best_epoch + 1})")
    print(f"  Best Validation R²:    {metrics.best_val_r2:.4f}")
    print(f"  Test Loss (MSE):       {test_loss:.4f}")
    print(f"  Test MAE:              {test_mae:.4f}")
    print(f"  Test R²:               {test_r2:.4f}")
    
    # Save metrics and generate plots
    print("\nGenerating training visualizations...")
    metrics.save_metrics()
    metrics.plot_metrics()
    
    summary = {
        'total_epochs': epoch + 1,
        'best_epoch': metrics.best_epoch + 1,
        'best_val_loss': metrics.best_val_loss,
        'best_val_r2': metrics.best_val_r2,
        'test_loss': test_loss,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'dataset_splits': {
            'train': train_size,
            'val': val_size,
            'test': test_size
        },
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'initial_lr': LR,
            'gamma': gamma,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'loss_fn': 'MSELoss'
        }
    }
    
    with open(run_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results saved to: {run_dir}")
    print(f"  - Metrics: {run_dir / 'regression_metrics.json'}")
    print(f"  - Plots: {run_dir / 'regression_metrics.png'}")
    print(f"  - Best model: {checkpoint_dir / 'best_value_model_state_dict.pt'}")
    print(f"  - Summary: {run_dir / 'training_summary.json'}")
    
    return {
        'network': network,
        'metrics': metrics,
        'test_loss': test_loss,
        'test_r2': test_r2,
        'run_dir': run_dir
    }
