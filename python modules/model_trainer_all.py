# ==============================================================================
# Global multi-country experiment runner - supports single and batch country experiments A, B, C
# ==============================================================================
import math
import contextlib
import io
import os
import json
import hashlib
import pickle
import tempfile
import pandas as pd                           # Data processing and analysis library
import numpy as np                            # Numerical computing library, provides efficient array operations
import lightgbm as lgb                        # LightGBM gradient boosting framework for tabular data
from sklearn.multioutput import MultiOutputRegressor  # Scikit-learn multi-output regression wrapper
from sklearn.model_selection import train_test_split  # Dataset splitting tool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Evaluation metrics
from sklearn.preprocessing import StandardScaler # Feature standardization tool
import torch                                  # PyTorch deep learning framework
import torch.nn as nn                         # PyTorch neural network module
import torch.optim as optim                   # PyTorch optimizer module
from torch.utils.data import TensorDataset, DataLoader  # PyTorch data loading tools
import optuna                                 # Automatic hyperparameter optimization library
from tqdm import tqdm                         # Progress bar display library (force text mode to avoid Notebook Widget issues)
import copy                                   # Python deep copy module
import warnings
import time                                   # Time statistics

# --- Optional dependency: Resource statistics ---
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

# --- Optional dependencies (auto-skip related models if not available) ---
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    from catboost import CatBoostRegressor  # type: ignore
except ImportError:  # pragma: no cover
    CatBoostRegressor = None

try:
    from pytorch_tabnet.tab_model import TabNetRegressor  # type: ignore
except ImportError:  # pragma: no cover
    TabNetRegressor = None

# --- Global settings ---
optuna.logging.set_verbosity(optuna.logging.INFO)  # Set Optuna log level to INFO, show tuning progress
warnings.filterwarnings('ignore')                      # Ignore warnings to keep output clean


@contextlib.contextmanager
def suppress_lightgbm_output():
    """Suppress LightGBM warnings during fit."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield

# --- NaN-aware loss and evaluation functions ---
import torch.nn.functional as F

class NaNAwareMSELoss(nn.Module):
    """
    MSE loss function supporting NaN in target variables
    
    This loss function intelligently handles NaN values in targets, computing loss only for valid values,
    perfectly adapted to real-world carbon emission data with partial missing target values.
    """
    def __init__(self):
        super(NaNAwareMSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        """
        Forward pass to compute loss
        
        Args:
            predictions: Model predictions, shape (batch_size, n_targets)
            targets: Ground truth targets, shape (batch_size, n_targets), may contain NaN
            
        Returns:
            MSE loss computed only on valid values
        """
        # Create valid value mask: True for non-NaN values
        valid_mask = ~torch.isnan(targets)
        
        # If all values in current batch are NaN, return zero loss
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        # Compute MSE loss only for valid values
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        return F.mse_loss(valid_predictions, valid_targets)

def nan_aware_r2_score(y_true, y_pred, multioutput='uniform_average'):
    """
    R² score calculation function supporting NaN
    
    Args:
        y_true: Ground truth, may contain NaN
        y_pred: Predictions
        multioutput: Multi-output handling method
        
    Returns:
        R² score with intelligent NaN handling
    """
    if y_true.ndim == 1:
        # Single target case
        valid_mask = ~np.isnan(y_true)
        if valid_mask.sum() < 2:  # At least 2 valid samples required to compute R²
            return np.nan
        return r2_score(y_true[valid_mask], y_pred[valid_mask])
    
    else:
        # Multi-target case
        r2_scores = []
        for i in range(y_true.shape[1]):
            valid_mask = ~np.isnan(y_true[:, i])
            if valid_mask.sum() >= 2 and len(np.unique(y_true[valid_mask, i])) > 1:
                r2_i = r2_score(y_true[valid_mask, i], y_pred[valid_mask, i])
                r2_scores.append(r2_i)
        
        if multioutput == 'uniform_average':
            return np.mean(r2_scores) if r2_scores else np.nan
        else:
            return r2_scores

# --- FT-Transformer and LoRA Components ---
class NumericalFeatureTokenizer(nn.Module):
    """Map numerical features to token sequence (with CLS)。"""

    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        token_emb = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, token_emb], dim=1)


class LoRAAdapter(nn.Module):
    """Simple LoRA module, simulate incremental weights with low-rank updates。"""

    def __init__(self, d_model, rank=8, alpha=16):
        super().__init__()
        rank = max(1, rank)
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.zeros(d_model, rank))
        self.B = nn.Parameter(torch.zeros(rank, d_model))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = x @ self.A @ self.B
        return x + update * self.scale


class FTTransformerEncoderLayer(nn.Module):
    """FT-Transformer encoder layer, supports LoRA adapter。"""

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1,
                 use_lora=False, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.use_lora = use_lora
        self.lora_attn = LoRAAdapter(d_model, lora_rank, lora_alpha) if use_lora else None
        self.lora_ffn = LoRAAdapter(d_model, lora_rank, lora_alpha) if use_lora else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        if self.use_lora:
            attn_out = self.lora_attn(attn_out)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        if self.use_lora:
            ffn_out = self.lora_ffn(ffn_out)
        x = self.norm2(x + ffn_out)
        return x


class FTTransformerModel(nn.Module):
    """Lightweight FT-Transformer implementation, supports LoRA mode。"""

    def __init__(self, n_features, n_targets, d_model=128, num_layers=4,
                 nhead=8, dim_feedforward=256, dropout=0.1,
                 use_lora=False, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_model)
        self.layers = nn.ModuleList([
            FTTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            ) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for layer in self.layers:
            tokens = layer(tokens)
        cls_repr = tokens[:, 0, :]
        return self.head(cls_repr)


def train_ft_transformer(model, train_loader, val_loader, device,
                         epochs=50, lr=1e-3, weight_decay=1e-4,
                         patience=10, checkpoint_path: str | None = None,
                         resume: bool = True):
    """Generic FT-Transformer training loop with early stopping and resumable checkpoints。"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = NaNAwareMSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0
    start_epoch = 0

    # Resume: Restore from checkpoint
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt.get('model_state', model.state_dict()))
            optimizer.load_state_dict(ckpt.get('optimizer_state', optimizer.state_dict()))
            best_state = ckpt.get('best_state', best_state)
            best_loss = ckpt.get('best_loss', best_loss)
            patience_counter = ckpt.get('patience_counter', 0)
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            print(f"    ⏭️ epoch resumed: {os.path.basename(checkpoint_path)} (from epoch {start_epoch})")
        except Exception as e:
            print(f"    ⚠️ epoch checkpoint read failed, will train from scratch: {e}")

    for epoch in range(start_epoch, epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        preds, targets = evaluate_pytorch(model, val_loader, device)
        loss = np.nanmean((preds - targets) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # epoch checkpoint
        if checkpoint_path:
            try:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_state': best_state,
                    'best_loss': best_loss,
                    'patience_counter': patience_counter,
                }, checkpoint_path)
            except Exception as e:
                print(f"    ⚠️ epoch checkpoint write failed (training continues): {e}")

    model.load_state_dict(best_state)
    return model

# --- PyTorch MLP Model Definition (original architecture) ---
class MLPNet(nn.Module):
    """
    Multi-layer perceptron class designed for multi-target regression on tabular data
    
    Architecture features:
    - Uses LayerNorm instead of BatchNorm, better suited for small batches and tabular data
    - Supports multi-target output (predict multiple carbon emission metrics simultaneously)
    - Flexible hidden layer architecture configuration
    """
    
    def __init__(self, n_features, n_targets, hidden_layers, dropout_rate):
        """
        Initialize MLP network structure
        
        Args:
            n_features (int): Number of input features
            n_targets (int): Number of output targets (e.g., 6 different carbon emission metrics)
            hidden_layers (list): List of neurons per hidden layer, e.g.[512, 256]
            dropout_rate (float): Dropout regularization rate to prevent overfitting
        """
        super(MLPNet, self).__init__()          # Call parent class nn.Module initialization
        layers = []                             # Create empty list to store network layers
        input_dim = n_features                  # Set first layer input dimension to feature count
        
        # Build hidden layer sequence
        for hidden_dim in hidden_layers:        # Iterate over each hidden layer neuron count
            layers.append(nn.Linear(input_dim, hidden_dim))    # Add linear transformation layer: y = xW^T + b
            layers.append(nn.LayerNorm(hidden_dim))            # Add layer normalization: normalize all features for each sample
            layers.append(nn.ReLU())                           # Add ReLU activation: max(0, x), introduce nonlinearity
            layers.append(nn.Dropout(dropout_rate))            # Add Dropout layer: randomly zero out neurons during training
            input_dim = hidden_dim              # Update input dimension for next layer
        
        layers.append(nn.Linear(input_dim, n_targets))         # Add output layer: linear transformation to target dimension
        self.model = nn.Sequential(*layers)                    # Combine all layers into a sequential container

    def forward(self, x):
        """
        Forward propagation function, defines data flow path in network
        
        Args:
            x (torch.Tensor): Input feature tensor, shape(batch_size, n_features)
            
        Returns:
            torch.Tensor: Prediction tensor, shape(batch_size, n_targets)
        """
        return self.model(x)                    # Pass input data through entire network

# --- PyTorch Training and Evaluation Helper Functions (Modified - NaN Support) ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Execute one complete training epoch - supports NaN-aware training
    
    Args:
        model (nn.Module): PyTorch model to train
        dataloader (DataLoader): Batch data loader for training
        optimizer (optim.Optimizer): Optimizer for updating model parameters based on gradients
        criterion (nn.Module): Loss function (should be NaN-aware)
        device (torch.device): Compute device (e.g., 'cpu' or 'cuda:0')
        
    Returns:
        float: Average loss for this epoch
    """
    model.train()                              # Set model to training mode
    total_loss = 0                             # Initialize cumulative loss
    valid_batches = 0                          # Count valid batches
    
    # Iterate over each batch in data loader
    for X_batch, y_batch in dataloader:
        # Move input and target data to specified compute device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Check if current batch has valid target values
        if torch.isnan(y_batch).all():
            continue  # Skip batch with all NaN
        
        optimizer.zero_grad()                  # Zero out previous gradients
        outputs = model(X_batch)               # Forward pass: compute predictions through model
        loss = criterion(outputs, y_batch)     # Compute loss (NaN-aware)
        
        # Check if loss is valid
        if torch.isnan(loss) or loss.item() == 0:
            continue  # Skip invalid loss
            
        loss.backward()                        # Backpropagation: compute gradients
        optimizer.step()                       # Update params: adjust weights based on gradients
        total_loss += loss.item()              # Accumulate loss value
        valid_batches += 1                     # Increment valid batch count
    
    return total_loss / valid_batches if valid_batches > 0 else 0.0

def evaluate_pytorch(model, dataloader, device):
    """
    Evaluate PyTorch model performance on given dataset
    
    Args:
        model (nn.Module): Trained PyTorch model
        dataloader (DataLoader): Batch data loader for evaluation
        device (torch.device): Compute device
        
    Returns:
        tuple: (Predictions matrix, Ground truth matrix), both as numpy arrays
    """
    model.eval()                               # Set model to evaluation mode (disable dropout, freeze batch norm)
    all_preds, all_targets = [], []            # Initialize lists to store predictions and targets from all batches
    
    with torch.no_grad():                      # Disable gradient computation (save memory and compute time)
        for X_batch, y_batch in dataloader:   # Iterate over each evaluation batch
            X_batch = X_batch.to(device)       # Move input to the compute device
            preds = model(X_batch)             # Get predictions through model
            all_preds.append(preds.cpu().numpy())     # Move predictions from GPU to CPU and convert to numpy array
            all_targets.append(y_batch.numpy())       # Convert ground truth to numpy array
    
    return np.vstack(all_preds), np.vstack(all_targets)  # Vertically concatenate results from all batches


def nan_aware_r2_score(y_true, y_pred, multioutput='uniform_average'):
    """
    R² score calculation function supporting NaN，Reference model_trainer_multiMLP_ynan.py
    
    Args:
        y_true: Ground truth, may contain NaN
        y_pred: Predictions
        multioutput: Multi-output handling method
        
    Returns:
        R² score with intelligent NaN handling
    """
    if y_true.ndim == 1:
        # Single target case
        valid_mask = ~np.isnan(y_true)
        if valid_mask.sum() < 2:  # Need at least 2 valid values
            return np.nan
        return r2_score(y_true[valid_mask], y_pred[valid_mask])
    
    else:
        # Multi-target case
        r2_scores = []
        for i in range(y_true.shape[1]):
            valid_mask = ~np.isnan(y_true[:, i])
            if valid_mask.sum() >= 2:  # Need at least 2 valid values
                r2 = r2_score(y_true[valid_mask, i], y_pred[valid_mask, i])
                r2_scores.append(r2)
        
        if multioutput == 'uniform_average':
            return np.mean(r2_scores) if r2_scores else np.nan
        else:
            return r2_scores


# --- Redesigned Global Experiment Runner ---
class GlobalExperimentRunner:
    """
    Global multi-country experiment runner - supports single and batch country runs
    
    Supports three experiment modes:
    - Experiment A: Country Self-Modeling - train and test using single country data
    - Experiment B: Global Baseline - train on global data, test on single country
    - Experiment C: Transfer Learning - global pre-training + single country fine-tuning
    
    Core design principles:
    - Unified data interface and evaluation standards
    - Flexible source and target domain configuration
    - NaN-aware data processing
    - Support single country or batch country runs
    """
    
    def __init__(
        self,
        full_data,
        features,
        targets,
        random_state=42,
        developed_countries=None,
        split_mode: str = "time",
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool | None = None,
        dataloader_persistent_workers: bool = False,
        dataloader_prefetch_factor: int | None = 2,
    ):
        """
        Initialize global experiment runner
        
        Args:
            full_data (pd.DataFrame): Full dataset across all countries; must contain a 'loc' column.
            features (list): List of feature column names
            targets (list): List of target variable column names
            random_state (int): Random seed for reproducible results
        """
        self.full_data = full_data.copy()       # Deep copy original data to avoid accidental modification
        self.features = features                # List of feature column names
        self.targets = targets                  # List of target variable column names
        self.random_state = random_state        # Store random seed
        self.developed_countries = set(developed_countries or [])
        # split_mode:
        # - "time": fiscalyear-based split for target-country evaluation (recommended, leakage-safe)
        # - "random": random train/val/test split (legacy, not recommended for benchmarking)
        self.split_mode = split_mode
        # Injected by run_global_experiments for epoch-level resume
        self._epoch_checkpoint_root = None

        # In-memory caches (trade RAM for speed). Safe because inputs are immutable for a runner instance.
        self._country_data_cache = {}
        self._source_data_cache = {}
        self._global_data_cache = {}

        # DataLoader config (for PyTorch-based experiments)
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        
        # Device detection: prefer GPU acceleration, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_info = "GPU" if torch.cuda.is_available() else "CPU"  # Get device info string
        print(f"🌍 GlobalExperimentRunner initialized. Compute device: {device_info}")
        
        # Get all country list and sort for result consistency
        self.all_countries = sorted(self.full_data['loc'].unique())
        print(f"📊 Found countries/regions")
        
        # Data quality pre-analysis: provide basis for subsequent country filtering
        self._analyze_global_data_quality()

    def _build_loader_kwargs(self):
        num_workers = int(self.dataloader_num_workers or 0)
        pin_memory = self.dataloader_pin_memory
        if pin_memory is None:
            pin_memory = (self.device.type == "cuda")

        kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            kwargs['persistent_workers'] = bool(self.dataloader_persistent_workers)
            if self.dataloader_prefetch_factor is not None:
                kwargs['prefetch_factor'] = int(self.dataloader_prefetch_factor)
        return kwargs
    
    def _analyze_global_data_quality(self):
        """
        Analyze global data quality, generate quality report for each country
        
        This method will:
        1. Calculate total samples for each country
        2. Calculate valid samples for each country (both targets and features non-null)
        3. Calculate data completeness rate
        4. Sort by sample size and display top 10
        """
        print("🔍 Analyzing global data quality...")
        
        country_quality = []  # Store data quality info for each country
        
        for country in self.all_countries:     # Iterate over all countries
            country_data = self.full_data[self.full_data['loc'] == country]  # Filter data for single country
            total_samples = len(country_data)   # Calculate total samples
            
            # Calculate valid samples: all target variables present and features complete
            label_complete_mask = country_data[self.targets].notnull().all(axis=1)
            feature_complete_mask = country_data[self.features].notnull().all(axis=1)
            usable_mask = label_complete_mask & feature_complete_mask
            label_complete = label_complete_mask.sum()
            usable_samples = usable_mask.sum()
            
            # Add country quality info to list
            country_quality.append({
                'country': country,
                'total': total_samples,
                'label_complete': label_complete,
                'usable': usable_samples,
                'usable_rate': (usable_samples / total_samples * 100) if total_samples > 0 else 0
            })
        
        # Sort by usable samples (descending)
        country_quality.sort(key=lambda x: x['usable'], reverse=True)
        
        # Display top 10 countries by sample size
        print("📈 Top 10 countries by sample size:")
        for i, cq in enumerate(country_quality[:10], 1):
            print(f"  {i:2d}. {cq['country']}: {cq['usable']:,}valid samples (all labels) ({cq['usable_rate']:.1f}%)")
        
        # Store results as dict for easy lookup
        self.country_quality = {cq['country']: cq for cq in country_quality}
    
    def _display_data_splits(self, experiment_type, country, **split_info):
        """
        Display experiment dataset split with detailed data source info
        
        Args:
            experiment_type (str): Experiment type ('A', 'B', 'C')
            country (str): Target country code
            **split_info: Keyword args for dataset split info
        """
        exp_names = {
            'A': 'Country Self-Modeling',
            'B': 'Global Baseline', 
            'B_dev': 'Developed Countries Baseline',
            'C': 'Transfer Learning',
            'C_dev': 'Transfer Learning (Developed)',
            'D': 'LoRA Transfer Learning',
            'D_dev': 'LoRA Transfer Learning (Developed)',
            'D_full': 'Full FT Transfer Learning',
            'D_full_dev': 'Full FT Transfer Learning (Developed)'
        }

        print(f"    📊 experiment{experiment_type}({exp_names.get(experiment_type, 'Unknown')}) - {country} Dataset split and sources:")

        if experiment_type == 'A':
            # Experiment A: Single country data split
            train_size = split_info.get('train_size', 0)
            test_size = split_info.get('test_size', 0)
            val_size = split_info.get('val_size', 0)
            total_usable = split_info.get('total_usable', 0)
            
            print(f"      � Data source strategy: Use only target country {country} local data")
            print(f"      �📈 {country} Total usable samples: {total_usable:,}")
            print(f"      🏋️ Training set: {train_size:,} samples ({train_size/total_usable*100:.1f}%) - Source: {country} country data")
            if val_size > 0:
                print(f"      ✅ validation set: {val_size:,} samples ({val_size/total_usable*100:.1f}%) - Source: {country} country data")
            print(f"      🎯 test set: {test_size:,} samples ({test_size/total_usable*100:.1f}%) - Source: {country} country data")
            print(f"      💡 Note: All datasets from target country, testing in-domain generalization")
            
        elif experiment_type in {'B', 'B_dev'}:
            # Experiment B: Global/Developed countries train, single country test
            global_train_size = split_info.get('global_train_size', 0)
            global_val_size = split_info.get('global_val_size', 0)
            target_test_size = split_info.get('target_test_size', 0)
            global_total = split_info.get('global_total', 0)
            source_desc = 'global data' if experiment_type == 'B' else 'developedcountry data'
            
            print(f"      📍 Data source strategy: {source_desc}train -> target country test")
            print(f"      🌍 Source domain training data (excluding{country}):")
            print(f"        🏋️ Training set: {global_train_size:,} samples ({global_train_size/global_total*100:.1f}%) - Source: {source_desc} (excluding{country})")
            print(f"        ✅ validation set: {global_val_size:,} samples ({global_val_size/global_total*100:.1f}%) - Source: {source_desc} (excluding{country})")
            print(f"      🎯 Target test data:")
            print(f"        🎯 test set: {target_test_size:,} samples - Source: {country} country data")
            print(f"      💡 description: use{source_desc}train, test cross-domain generalization on target country")
            
        elif experiment_type in {'C', 'C_dev', 'D', 'D_dev', 'D_full', 'D_full_dev'}:
            # Experiment C/D: Transfer Learning
            global_pretrain_size = split_info.get('global_pretrain_size', 0)
            target_train_size = split_info.get('target_train_size', 0)
            target_val_size = split_info.get('target_val_size', 0)
            target_test_size = split_info.get('target_test_size', 0)
            target_total = split_info.get('target_total', 0)
            source_desc = 'global data' if experiment_type in {'C', 'D', 'D_full'} else 'developedcountry data'
            if experiment_type.startswith('C'):
                method_desc = 'MLP pre-training + fine-tuning'
            elif experiment_type.startswith('D_full'):
                method_desc = 'FT-Transformer Full FT pre-training + fine-tuning'
            else:
                method_desc = 'FT-Transformer LoRA pre-training + fine-tuning'
            
            print(f"      📍 Data source strategy: {source_desc}pre-training → target countryfine-tuning ({method_desc})")
            print("      🌍 Phase 1 — source-domain pre-training:")
            print(f"        🏗️ Pre-training set: {global_pretrain_size:,} samples - Source: {source_desc} (excluding {country})")
            print(f"      🎯 Phase 2 — {country} fine-tuning and testing:")
            print(f"        📈 Total usable samples: {target_total:,} - Source: {country} country data")
            print(f"        🏋️ fine-tuningTraining set: {target_train_size:,} samples ({target_train_size/target_total*100:.1f}%) - Source: {country} country data")
            if target_val_size > 0:
                print(f"        ✅ fine-tuningvalidation set: {target_val_size:,} samples ({target_val_size/target_total*100:.1f}%) - Source: {country} country data")
            print(f"        🎯 test set: {target_test_size:,} samples ({target_test_size/target_total*100:.1f}%) - Source: {country} country data")
            print(f"      💡 Description: pre-train on {source_desc}, then fine-tune on target-country data to adapt to local features.")
    
    def _prepare_country_data(self, country):
        """
        Prepare cleaned modeling data for a given country (no-fill policy).
        
        Args:
            country (str): country code (e.g., 'CHN', 'USA')
            
        Returns:
            tuple or None: (X, y, years, sample_count, original_count) if data is sufficient; otherwise return None
                - X (np.ndarray): feature matrix, shape (n_samples, n_features)
                - y (np.ndarray): target matrix, shape (n_samples, n_targets), with no NaN values
                - years (np.ndarray): fiscal-year array (aligned with X/y)
                - sample_count (int): validsamplescount
                - original_count (int): original sample count (before cleaning)
        """
        cached = self._country_data_cache.get(country)
        if cached is not None:
            return cached

        # Filter data for the target country
        country_data = self.full_data[self.full_data['loc'] == country].copy()
        original_count = len(country_data)  # original sample count
        
        # checkyesnohasdata
        if original_count == 0:
            return None
        
        # No-fill policy: keep only rows where all targets and all features are present
        all_targets_valid_mask = country_data[self.targets].notnull().all(axis=1)  # all targets are present
        feature_complete_mask = country_data[self.features].notnull().all(axis=1)  # allall feature variables are notempty
        usable_mask = all_targets_valid_mask & feature_complete_mask               # combined availability mask
        
        usable_count = usable_mask.sum()       # usable sample count
        if usable_count < 10:                  # require at least 10 samples to train a model
            return None
        
        # Extract cleaned data
        clean_data = country_data[usable_mask]
        X = clean_data[self.features].values.astype(np.float32)  # feature matrix，convertasfloat32save memory
        y = clean_data[self.targets].values.astype(np.float32)   # targetmatrix，convertasfloat32save memory
        years = clean_data["fiscalyear"].values
        
        # extract industryinfo（ifexists）
        sectors = clean_data["GICSSector"].values if "GICSSector" in clean_data.columns else None
        
        # verify：Ensure noNaN（nonefillcore of the strategy）
        assert not np.isnan(X).any(), "Feature matrix should not haveNaN"
        assert not np.isnan(y).any(), "Target matrix should not contain NaN (no-fill policy)."
        
        result = (X, y, years, usable_count, original_count, sectors)
        self._country_data_cache[country] = result
        return result
    
    def _prepare_global_data(self, exclude_country=None):
        """
        Prepare global data (optionally excluding a country) under the no-fill policy.
        
        Args:
            exclude_country (str, optional): country code to exclude (used in Experiments B and C)
            
        Returns:
            tuple: (X_global, y_global, sample_count)
                - X_global (np.ndarray): globalfeature matrix
                - y_global (np.ndarray): globaltargetmatrix，noneNaN
                - sample_count (int): number of usable global samples
        """
        cache_key = exclude_country
        cached = self._global_data_cache.get(cache_key)
        if cached is not None:
            return cached

        # Filter global data, optionally excluding the target country
        if exclude_country:
            global_data = self.full_data[self.full_data['loc'] != exclude_country].copy()
        else:
            global_data = self.full_data.copy()                                            # usealldata
        
        # No-fill policy: keep only rows where all targets and all features are present
        all_targets_valid_mask = global_data[self.targets].notnull().all(axis=1)   # all targets are present
        feature_complete_mask = global_data[self.features].notnull().all(axis=1)   # allall feature variables are notempty
        usable_mask = all_targets_valid_mask & feature_complete_mask               # combined availability mask
        
        usable_count = usable_mask.sum()
        if usable_count < 100:  # global data requires sufficient samples
            print(f"    ⚠️ globalvalidsamplesinsufficient: {usable_count} < 100")
            return None, None, 0
        
        # Extract cleaned data
        clean_global_data = global_data[usable_mask]
        X_global = clean_global_data[self.features].values.astype(np.float32)
        y_global = clean_global_data[self.targets].values.astype(np.float32)
        
        # Verify: ensure no NaN values (core of the no-fill policy)
        assert not np.isnan(X_global).any(), "globalFeature matrix should not haveNaN"
        assert not np.isnan(y_global).any(), "Global target matrix should not contain NaN (no-fill policy)."
        
        result = (X_global, y_global, usable_count)
        self._global_data_cache[cache_key] = result
        return result

    def _prepare_source_data(self, target_country, source_mode='all'):
        """Prepare cross-country training data by source-domain mode (all/dev)."""
        cache_key = (target_country, source_mode)
        cached = self._source_data_cache.get(cache_key)
        if cached is not None:
            return cached

        data = self.full_data.copy()

        if source_mode == 'dev':
            if not self.developed_countries:
                raise ValueError("developed_countries is not provided; cannot run *_dev experiments.")
            data = data[data['loc'].isin(self.developed_countries)]

        if target_country is not None:
            data = data[data['loc'] != target_country]

        all_targets_valid_mask = data[self.targets].notnull().all(axis=1)
        feature_complete_mask = data[self.features].notnull().all(axis=1)
        usable_mask = all_targets_valid_mask & feature_complete_mask
        usable_count = usable_mask.sum()

        if usable_count < 50:
            result = (None, None, 0)
            self._source_data_cache[cache_key] = result
            return result

        clean_data = data[usable_mask]
        X = clean_data[self.features].values.astype(np.float32)
        y = clean_data[self.targets].values.astype(np.float32)
        result = (X, y, usable_count)
        self._source_data_cache[cache_key] = result
        return result

    def _split_local_data(self, X, y, test_size=0.15, val_size=0.15):
        """Unified training/validation/testing split logic."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _split_local_data_time(self, X, y, years):
        """
        Time-aware split (to avoid temporal leakage):
        - train: all years except the last 2 years
        - val: the second-to-last year
        - test: the last year
        If the number of unique years is insufficient (<3), fall back to a random split.
        """
        unique_years = sorted(pd.unique(years))
        if len(unique_years) < 3:
            return self._split_local_data(X, y)

        train_years = set(unique_years[:-2])
        val_years = {unique_years[-2]}
        test_years = {unique_years[-1]}

        train_mask = np.isin(years, list(train_years))
        val_mask = np.isin(years, list(val_years))
        test_mask = np.isin(years, list(test_years))

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # If any split becomes empty (extreme case), fall back to a random split.
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            return self._split_local_data(X, y)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _handle_nan_targets(self, y):
        """
        Handle NaN values in targets (no-fill policy version).
        
        Under the no-fill policy, the training data should not contain NaN targets because we have already
        filtered to samples with valid values for all target variables during data preparation.
        
        Args:
            y (np.ndarray): target matrix; should not contain NaN
            
        Returns:
            np.ndarray: verifytarget aftermatrix
        """
        # Verify: under the no-fill policy, training targets should not contain NaN.
        if np.isnan(y).any():
            raise ValueError(
                "Under the no-fill policy, training targets should not contain NaN values. "
                "Please check the data-preparation logic."
            )
        
        return y.copy()  # returncopy to avoid accidentalmodifyoriginal data
    
    def _get_evaluation_scores(self, y_true, y_pred, model_name, country, experiment_type):
        """
        Compute evaluation metrics separately for each target variable (R², MAE, RMSE).
        
        Args:
            y_true (np.ndarray): ground-truth value matrix
            y_pred (np.ndarray): Predictionsmatrix
            model_name (str): model-name identifier
            country (str): country code
            experiment_type (str): experiment-type identifier
            
        Returns:
            dict: dictionary containing all evaluation metrics
        """
        # initializeresultsdictionary，containsbasicinfo
        results = {
            'model': model_name,
            'country': country,
            'experiment_type': experiment_type
        }
        
        avg_r2 = 0          # for calculating averageR²
        valid_targets = 0   # logvalidtarget variablecount
        
        # Calculate separately for each target variablemetrics
        for i, target in enumerate(self.targets):
            # Identify valid samples (excluding NaN).
            valid_mask = ~(np.isnan(y_true[:, i]) | np.isnan(y_pred[:, i]))
            
            # Compute metrics only if there are enough valid samples and y_true has non-zero variance.
            if valid_mask.sum() > 1 and len(np.unique(y_true[valid_mask, i])) > 1:
                # Compute per-target metrics.
                r2 = r2_score(y_true[valid_mask, i], y_pred[valid_mask, i])          # R²coefficient of determination
                mae = mean_absolute_error(y_true[valid_mask, i], y_pred[valid_mask, i])  # averageabsoluteerror
                rmse = np.sqrt(mean_squared_error(y_true[valid_mask, i], y_pred[valid_mask, i]))  # root mean square error
                
                # Store per-target metrics.
                results[f'R2_{target}'] = r2
                results[f'MAE_{target}'] = mae
                results[f'RMSE_{target}'] = rmse
                
                avg_r2 += r2        # accumulateR²
                valid_targets += 1  # validtarget count plus1
            else:
                # Invalid target: set metrics to NaN
                results[f'R2_{target}'] = np.nan
                results[f'MAE_{target}'] = np.nan
                results[f'RMSE_{target}'] = np.nan
        
        # calculateaverageR²
        results['R2_Average'] = avg_r2 / valid_targets if valid_targets > 0 else np.nan
        results['Valid_Targets'] = valid_targets
        
        return results

    def run_experiment_A(self, country, n_trials=10, tabnet_epochs=50, tabnet_patience=10,
                          ft_epochs=80, ft_lr=1e-3, ft_dropout=0.1,
                          lgbm_n_estimators_min=100, lgbm_n_estimators_max=200,
                          xgb_n_estimators_min=100, xgb_n_estimators_max=200,
                          catboost_iterations_min=100, catboost_iterations_max=200,
                          optuna_timeout_seconds: int | None = None):
        """
        Experiment A: local model competition (A* gold-standard baseline).

        - LightGBM / XGBoost / CatBoost: Optuna tuning; friendly to small samples
        - TabNet: sparse-attention tabular network for processed features
        - FT-Transformer: transformer-style tabular model

        After training all candidates locally, we rank by validation-set R² and take the winner as A*.
        """
        print(f"  🏠 Experiment A: {country} local model competition (A*)")

        country_result = self._prepare_country_data(country)
        if country_result is None:
            return {
                'model': f'GBDT-A-{country}',
                'country': country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': 'Country_Self_Modeling',
                'experiment_code': 'A'
            }

        try:
            X, y, years, sample_count, original_count, sectors = country_result
            if self.split_mode == "time":
                X_train, y_train, X_val, y_val, X_test, y_test = self._split_local_data_time(X, y, years)
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = self._split_local_data(X, y)

            self._display_data_splits(
                experiment_type='A',
                country=country,
                train_size=X_train.shape[0],
                val_size=X_val.shape[0],
                test_size=X_test.shape[0],
                total_usable=sample_count,
                original_size=original_count
            )

            candidate_results = []
            
            # --- 1. LightGBM with Optuna ---
            print("    🔍 Tuning LightGBM...")
            def lgbm_objective(trial):
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'random_state': self.random_state,
                    'verbosity': -1,
                    'n_estimators': trial.suggest_int('n_estimators', lgbm_n_estimators_min, lgbm_n_estimators_max),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'max_depth': trial.suggest_int('max_depth', 4, 16),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
                with suppress_lightgbm_output():
                    model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return nan_aware_r2_score(y_val, preds, multioutput='uniform_average')

            lgbm_study = optuna.create_study(direction='maximize')
            lgbm_study.optimize(
                lgbm_objective,
                n_trials=n_trials,
                timeout=optuna_timeout_seconds,
                show_progress_bar=True,
            )
            
            lgbm_best_params = lgbm_study.best_params.copy()
            lgbm_best_params['random_state'] = self.random_state
            lgbm_best_params['verbosity'] = -1
            lgbm_model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_best_params))
            X_train_full = np.vstack([X_train, X_val])
            y_train_full = np.vstack([y_train, y_val])
            with suppress_lightgbm_output():
                lgbm_model.fit(X_train_full, y_train_full)
            
            # calculatetraining setR2
            lgbm_train_pred = lgbm_model.predict(X_train_full)
            lgbm_train_r2 = nan_aware_r2_score(y_train_full, lgbm_train_pred, multioutput='uniform_average')
            
            lgbm_test_pred = lgbm_model.predict(X_test)
            lgbm_result = self._get_evaluation_scores(
                y_test, lgbm_test_pred,
                f'LightGBM-A-{country}', country, 'Country_Self_Modeling'
            )
            lgbm_result['validation_R2'] = lgbm_study.best_value
            lgbm_result['training_R2'] = lgbm_train_r2
            lgbm_result['candidate'] = 'LightGBM'
            lgbm_result['best_params'] = lgbm_best_params
            candidate_results.append(lgbm_result)
            print(f"      ✓ LightGBM: trainR²={lgbm_train_r2:.4f}, verifyR²={lgbm_study.best_value:.4f}, testingR²={lgbm_result['R2_Average']:.4f}")

            # --- 2. XGBoost with Optuna ---
            if xgb is not None:
                print("    🔍 Tuning XGBoost...")
                def xgb_objective(trial):
                    params = {
                        'objective': 'reg:squarederror',
                        'random_state': self.random_state,
                        'verbosity': 0,
                        'tree_method': 'hist',
                        'n_estimators': trial.suggest_int('n_estimators', xgb_n_estimators_min, xgb_n_estimators_max),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'max_depth': trial.suggest_int('max_depth', 4, 16),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                    }
                    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    return nan_aware_r2_score(y_val, preds, multioutput='uniform_average')

                xgb_study = optuna.create_study(direction='maximize')
                xgb_study.optimize(
                    xgb_objective,
                    n_trials=n_trials,
                    timeout=optuna_timeout_seconds,
                    show_progress_bar=True,
                )
                
                xgb_best_params = xgb_study.best_params.copy()
                xgb_best_params['random_state'] = self.random_state
                xgb_best_params['verbosity'] = 0
                xgb_best_params['tree_method'] = 'hist'
                xgb_model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_best_params))
                xgb_model.fit(X_train_full, y_train_full)
                
                # calculatetraining setR2
                xgb_train_pred = xgb_model.predict(X_train_full)
                xgb_train_r2 = nan_aware_r2_score(y_train_full, xgb_train_pred, multioutput='uniform_average')
                
                xgb_test_pred = xgb_model.predict(X_test)
                xgb_result = self._get_evaluation_scores(
                    y_test, xgb_test_pred,
                    f'XGBoost-A-{country}', country, 'Country_Self_Modeling'
                )
                xgb_result['validation_R2'] = xgb_study.best_value
                xgb_result['training_R2'] = xgb_train_r2
                xgb_result['candidate'] = 'XGBoost'
                xgb_result['best_params'] = xgb_best_params
                candidate_results.append(xgb_result)
                print(f"      ✓ XGBoost: trainR²={xgb_train_r2:.4f}, verifyR²={xgb_study.best_value:.4f}, testingR²={xgb_result['R2_Average']:.4f}")
            else:
                print("    ⚠️ XGBoost is not installed; skipping.")

            # --- 3. CatBoost with Optuna ---
            if CatBoostRegressor is not None:
                print("    🔍 Tuning CatBoost...")
                def catboost_objective(trial):
                    params = {
                        'loss_function': 'RMSE',
                        'random_seed': self.random_state,
                        'verbose': False,
                        'iterations': trial.suggest_int('iterations', catboost_iterations_min, catboost_iterations_max),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'depth': trial.suggest_int('depth', 4, 10),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
                        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0)
                    }
                    model = MultiOutputRegressor(CatBoostRegressor(**params))
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    return nan_aware_r2_score(y_val, preds, multioutput='uniform_average')

                catboost_study = optuna.create_study(direction='maximize')
                catboost_study.optimize(
                    catboost_objective,
                    n_trials=n_trials,
                    timeout=optuna_timeout_seconds,
                    show_progress_bar=True,
                )
                
                catboost_best_params = catboost_study.best_params.copy()
                catboost_best_params['random_seed'] = self.random_state
                catboost_best_params['verbose'] = False
                catboost_model = MultiOutputRegressor(CatBoostRegressor(**catboost_best_params))
                catboost_model.fit(X_train_full, y_train_full)
                
                # calculatetraining setR2
                catboost_train_pred = catboost_model.predict(X_train_full)
                catboost_train_r2 = nan_aware_r2_score(y_train_full, catboost_train_pred, multioutput='uniform_average')
                
                catboost_test_pred = catboost_model.predict(X_test)
                catboost_result = self._get_evaluation_scores(
                    y_test, catboost_test_pred,
                    f'CatBoost-A-{country}', country, 'Country_Self_Modeling'
                )
                catboost_result['validation_R2'] = catboost_study.best_value
                catboost_result['training_R2'] = catboost_train_r2
                catboost_result['candidate'] = 'CatBoost'
                catboost_result['best_params'] = catboost_best_params
                candidate_results.append(catboost_result)
                print(f"      ✓ CatBoost: trainR²={catboost_train_r2:.4f}, verifyR²={catboost_study.best_value:.4f}, testingR²={catboost_result['R2_Average']:.4f}")
            else:
                print("    ⚠️ CatBoost is not installed; skipping.")

            # --- 4/5. Deep tabular candidates (optional in QUICK_RUN) ---
            need_scaled = False
            if TabNetRegressor is not None and tabnet_epochs and tabnet_epochs > 0:
                need_scaled = True
            if ft_epochs and ft_epochs > 0:
                need_scaled = True

            scaled = None
            scaled_y = None
            if need_scaled:
                scaler = StandardScaler().fit(X_train)
                scaled = {
                    'X_train': scaler.transform(X_train).astype(np.float32),
                    'X_val': scaler.transform(X_val).astype(np.float32),
                    'X_test': scaler.transform(X_test).astype(np.float32)
                }
                scaled_y = {
                    'y_train': y_train.astype(np.float32),
                    'y_val': y_val.astype(np.float32),
                    'y_test': y_test.astype(np.float32)
                }

            if TabNetRegressor is not None and tabnet_epochs and tabnet_epochs > 0:
                print("    🔍 train TabNet...")
                tabnet = TabNetRegressor(
                    n_d=64,
                    n_a=64,
                    n_steps=5,
                    gamma=1.5,
                    lambda_sparse=1e-4,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=1e-3),
                    seed=self.random_state,
                    verbose=0
                )
                tabnet.fit(
                    scaled['X_train'], scaled_y['y_train'],
                    eval_set=[(scaled['X_val'], scaled_y['y_val'])],
                    patience=tabnet_patience,
                    max_epochs=tabnet_epochs,
                    batch_size=1024,
                    virtual_batch_size=128,
                    num_workers=0,
                    drop_last=False
                )
                
                # calculatetraining setR2
                tabnet_train_pred = tabnet.predict(scaled['X_train'])
                tabnet_train_r2 = nan_aware_r2_score(scaled_y['y_train'], tabnet_train_pred, multioutput='uniform_average')
                
                tabnet_val_pred = tabnet.predict(scaled['X_val'])
                tabnet_val_r2 = nan_aware_r2_score(scaled_y['y_val'], tabnet_val_pred, multioutput='uniform_average')
                tabnet_test_pred = tabnet.predict(scaled['X_test'])
                tabnet_result = self._get_evaluation_scores(
                    scaled_y['y_test'], tabnet_test_pred,
                    f'TabNet-A-{country}', country, 'Country_Self_Modeling'
                )
                tabnet_result['validation_R2'] = tabnet_val_r2
                tabnet_result['training_R2'] = tabnet_train_r2
                tabnet_result['candidate'] = 'TabNet'
                candidate_results.append(tabnet_result)
                print(f"      ✓ TabNet: trainR²={tabnet_train_r2:.4f}, verifyR²={tabnet_val_r2:.4f}, testingR²={tabnet_result['R2_Average']:.4f}")
            elif TabNetRegressor is None:
                print("    ⚠️ pytorch-tabnet is not installed; skipping TabNet.")
            else:
                print("    ⏭️ TabNet skipped (tabnet_epochs<=0)")

            # --- 5. FT-Transformer ---
            if ft_epochs and ft_epochs > 0:
                print("    🔍 train FT-Transformer...")
                loader_kwargs = self._build_loader_kwargs()
                ft_model = FTTransformerModel(
                    n_features=X.shape[1],
                    n_targets=y.shape[1],
                    d_model=128,
                    num_layers=4,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=ft_dropout
                ).to(self.device)

                ft_train_loader = DataLoader(
                    TensorDataset(torch.from_numpy(scaled['X_train']), torch.from_numpy(scaled_y['y_train'])),
                    batch_size=256,
                    shuffle=True,
                    **loader_kwargs,
                )
                ft_val_loader = DataLoader(
                    TensorDataset(torch.from_numpy(scaled['X_val']), torch.from_numpy(scaled_y['y_val'])),
                    batch_size=256,
                    **loader_kwargs,
                )
                ft_test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(scaled['X_test']), torch.from_numpy(scaled_y['y_test'])),
                    batch_size=256,
                    **loader_kwargs,
                )

                # Epoch-level resume for the FT-Transformer candidate (if checkpoints are enabled).
                ckpt_path = None
                if self._epoch_checkpoint_root:
                    ckpt_path = os.path.join(self._epoch_checkpoint_root, f"{country}__A__ftt.pt")
                train_ft_transformer(
                    ft_model,
                    ft_train_loader,
                    ft_val_loader,
                    self.device,
                    epochs=ft_epochs,
                    lr=ft_lr,
                    checkpoint_path=ckpt_path,
                    resume=True,
                )
                
                # calculatetraining setR2
                ft_train_pred, ft_train_true = evaluate_pytorch(ft_model, ft_train_loader, self.device)
                ft_train_r2 = nan_aware_r2_score(ft_train_true, ft_train_pred, multioutput='uniform_average')
                
                ft_val_pred, ft_val_true = evaluate_pytorch(ft_model, ft_val_loader, self.device)
                ft_val_r2 = nan_aware_r2_score(ft_val_true, ft_val_pred, multioutput='uniform_average')
                ft_test_pred, ft_test_true = evaluate_pytorch(ft_model, ft_test_loader, self.device)
                ft_result = self._get_evaluation_scores(
                    ft_test_true, ft_test_pred,
                    f'FTT-A-{country}', country, 'Country_Self_Modeling'
                )
                ft_result['validation_R2'] = ft_val_r2
                ft_result['training_R2'] = ft_train_r2
                ft_result['candidate'] = 'FT-Transformer'
                candidate_results.append(ft_result)
                print(f"      ✓ FT-Transformer: trainR²={ft_train_r2:.4f}, verifyR²={ft_val_r2:.4f}, testingR²={ft_result['R2_Average']:.4f}")
            else:
                print("    ⏭️ FT-Transformer skipped (ft_epochs<=0)")

            # --- Select the final A* (gold-standard) model ---
            valid_candidates = [r for r in candidate_results if not np.isnan(r.get('validation_R2', np.nan))]
            if not valid_candidates:
                raise RuntimeError("All Experiment-A candidate models failed.")

            # Tag candidate results for saving
            for entry in candidate_results:
                entry['experiment_code'] = f"A_{entry.get('candidate', 'candidate')}"

            for entry in valid_candidates:
                entry['training_samples'] = sample_count
                entry['valid_training_samples'] = sample_count
                entry['data_efficiency'] = sample_count / original_count

            # Key point: select the winner using validation metrics only (to avoid test-set leakage).
            leaderboard = sorted(valid_candidates, key=lambda r: r['validation_R2'], reverse=True)
            best = leaderboard[0]

            print("    🏁 A* ranking (validation R²):")
            for rank, entry in enumerate(leaderboard, 1):
                print(f"      #{rank}. {entry['candidate']}: verifyR²={entry['validation_R2']:.4f}, testingR²={entry['R2_Average']:.4f}")

            best['model'] = f"A_star-{best['candidate']}-{country}"
            best['experiment_code'] = 'A_star'
            best['selected_candidate'] = best['candidate']
            best['selected_model_validation_R2'] = best['validation_R2']
            best['candidate_scores'] = [
                {
                    'candidate': entry['candidate'],
                    'model': entry['model'],
                    'validation_R2': entry['validation_R2'],
                    'test_R2': entry['R2_Average']
                }
                for entry in leaderboard
            ]
            best['candidate_results'] = candidate_results
            
            # Add sample-size and year metadata for downstream analysis.
            best['n_train'] = X_train.shape[0]
            best['n_val'] = X_val.shape[0]
            best['n_test'] = X_test.shape[0]
            best['n_total'] = sample_count
            best['years_min'] = int(years.min()) if len(years) > 0 else None
            best['years_max'] = int(years.max()) if len(years) > 0 else None
            best['n_years'] = len(np.unique(years)) if len(years) > 0 else 0
            best['is_developed'] = country in self.developed_countries if self.developed_countries else None
            
            # Add industry statistics (for industry-subset analysis).
            if sectors is not None:
                unique_sectors = np.unique(sectors[~pd.isnull(sectors)])
                sector_counts = pd.Series(sectors).value_counts()
                best['n_sectors'] = len(unique_sectors)
                best['dominant_sector'] = sector_counts.index[0] if len(sector_counts) > 0 else None
                best['dominant_sector_ratio'] = sector_counts.iloc[0] / len(sectors) if len(sector_counts) > 0 else None
                best['sector_distribution'] = sector_counts.to_dict()
            else:
                best['n_sectors'] = None
                best['dominant_sector'] = None
                best['dominant_sector_ratio'] = None
                best['sector_distribution'] = None

            print(
                f"    🏆 A* winner: {best['candidate']} "
                f"(val R²={best['validation_R2']:.4f}, test R²={best['R2_Average']:.4f})"
            )
            return best

        except Exception as e:
            print(f"    ❌ failure: {str(e)}")
            return {
                'model': f'A_star-{country}',
                'country': country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Country_Self_Modeling',
                'experiment_code': 'A'
            }

    def run_experiment_B(self, target_country, n_trials=25, source_mode='all',
                         lgbm_n_estimators=1200, lgbm_final_estimators=2000,
                         optuna_timeout_seconds: int | None = None):
        """Experiment B: global/developed-country baseline modeling."""
        exp_code = 'B_dev' if source_mode == 'dev' else 'B'
        exp_name = 'developedcountry' if source_mode == 'dev' else 'global'
        print(f"  🌍 experiment{exp_code}: {exp_name}→{target_country} baseline modeling")

        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'LGBM-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Global_Baseline' if source_mode == 'all' else 'Global_Baseline_Developed',
                'experiment_code': exp_code
            }

        X_target, y_target, years_target, target_samples, _, sectors = target_result

        X_source, y_source, source_samples = self._prepare_source_data(target_country, source_mode)
        if source_samples < 100:
            return {
                'model': f'LGBM-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_source_data',
                'experiment_type': 'Global_Baseline' if source_mode == 'all' else 'Global_Baseline_Developed',
                'experiment_code': exp_code
            }

        print(f"    📊 source domainsamples: {source_samples:,}, target test samples: {target_samples:,}")

        try:
            # modifyas80%train，20%verify，Test set uses target domain
            X_train, X_val, y_train, y_val = train_test_split(X_source, y_source, test_size=0.2, random_state=self.random_state)
            
            self._display_data_splits(
                experiment_type='B' if source_mode == 'all' else 'B_dev',
                country=target_country,
                global_train_size=X_train.shape[0],
                global_val_size=X_val.shape[0],
                target_test_size=target_samples,
                global_total=source_samples
            )

            def objective(trial):
                params = {
                    'objective': 'regression_l1',
                    'metric': 'rmse',
                    'random_state': self.random_state,
                    'n_estimators': lgbm_n_estimators,
                    'n_jobs': -1,
                    'verbosity': -1,
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 32, 256),
                    'max_depth': trial.suggest_int('max_depth', 5, 18),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
                with suppress_lightgbm_output():
                    model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return nan_aware_r2_score(y_val, preds, multioutput='uniform_average')

            study = optuna.create_study(direction='maximize')
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=optuna_timeout_seconds,
                show_progress_bar=True,
            )

            best_params = study.best_params
            best_params.update({'n_estimators': lgbm_final_estimators, 'random_state': self.random_state, 'n_jobs': -1, 'verbosity': -1})
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**best_params))
            
            # useallsource domainretrain with data
            X_source_full = np.vstack([X_train, X_val])
            y_source_full = np.vstack([y_train, y_val])
            with suppress_lightgbm_output():
                final_model.fit(X_source_full, y_source_full)

            # calculatetraining setR2 (source domain)
            y_pred_train = final_model.predict(X_source_full)
            train_r2 = nan_aware_r2_score(y_source_full, y_pred_train, multioutput='uniform_average')

            y_pred_target = final_model.predict(X_target)
            result = self._get_evaluation_scores(
                y_target, y_pred_target,
                f'LGBM-{exp_code}-{target_country}', target_country,
                'Global_Baseline' if source_mode == 'all' else 'Global_Baseline_Developed'
            )
            result['training_samples'] = source_samples
            result['test_samples'] = target_samples
            result['optuna_best_validation_R2'] = study.best_value
            result['training_R2'] = train_r2
            result['source_mode'] = source_mode
            result['experiment_code'] = exp_code
            
            # Add sample-size and year metadata for downstream analysis.
            result['n_source'] = source_samples
            result['n_train'] = X_train.shape[0]
            result['n_val'] = X_val.shape[0]
            result['n_test'] = target_samples
            result['n_total'] = target_samples
            result['years_min'] = int(years_target.min()) if len(years_target) > 0 else None
            result['years_max'] = int(years_target.max()) if len(years_target) > 0 else None
            result['n_years'] = len(np.unique(years_target)) if len(years_target) > 0 else 0
            result['is_developed'] = target_country in self.developed_countries if self.developed_countries else None
            
            # Add industry statistics (for industry-subset analysis).
            if sectors is not None:
                unique_sectors = np.unique(sectors[~pd.isnull(sectors)])
                sector_counts = pd.Series(sectors).value_counts()
                result['n_sectors'] = len(unique_sectors)
                result['dominant_sector'] = sector_counts.index[0] if len(sector_counts) > 0 else None
                result['dominant_sector_ratio'] = sector_counts.iloc[0] / len(sectors) if len(sector_counts) > 0 else None
            else:
                result['n_sectors'] = None
                result['dominant_sector'] = None
                result['dominant_sector_ratio'] = None

            print(f"    ✅ complete: trainR²={train_r2:.4f}, verifyR²={study.best_value:.4f}, testingR²={result['R2_Average']:.4f}")
            return result

        except Exception as e:
            print(f"    ❌ failure: {str(e)}")
            return {
                'model': f'LGBM-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Global_Baseline' if source_mode == 'all' else 'Global_Baseline_Developed',
                'experiment_code': exp_code
            }
    
    def run_experiment_C(
        self,
        target_country,
        source_mode='all',
        n_finetune_trials=25,
        stage2_trials: int = 0,
        stage2_min_val_r2: float = -float('inf'),
        pretrain_epochs=30,
        finetune_epochs=50,
        finetune_min_epochs: int = 10,
        finetune_patience: int = 10,
        use_pruner: bool = True,
        pruner_warmup_epochs: int = 5,
        pretrain_batch_size=1024,
        finetune_batch_size=64,
        eval_batch_size=128,
    ):
        """experimentC/C_dev: MLPtransfer learning（globalordevelopedcountrypre-training）。"""
        exp_code = 'C_dev' if source_mode == 'dev' else 'C'
        exp_name = 'developedcountry' if source_mode == 'dev' else 'global'
        print(f"  🔄 experiment{exp_code}: {exp_name}→{target_country} transfer learning")

        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'MLP-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Transfer_Learning' if source_mode == 'all' else 'Transfer_Learning_Developed',
                'experiment_code': exp_code
            }

        X_target, y_target, years_target, target_samples, _, sectors = target_result
        X_source, y_source, source_samples = self._prepare_source_data(target_country, source_mode)

        if source_samples < 100 or target_samples < 20:
            return {
                'model': f'MLP-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': 'Transfer_Learning' if source_mode == 'all' else 'Transfer_Learning_Developed',
                'experiment_code': exp_code
            }

        print(f"    📊 pre-trainingsamples: {source_samples:,}, fine-tuning samples: {target_samples:,}")

        try:
            pretrain_scaler = StandardScaler()
            X_source_scaled = pretrain_scaler.fit_transform(X_source).astype(np.float32)
            finetune_scaler = StandardScaler()
            X_target_scaled = finetune_scaler.fit_transform(X_target).astype(np.float32)

            global_dataset = TensorDataset(torch.from_numpy(X_source_scaled), torch.from_numpy(y_source))
            loader_kwargs = self._build_loader_kwargs()
            global_loader = DataLoader(
                global_dataset,
                batch_size=pretrain_batch_size,
                shuffle=True,
                **loader_kwargs,
            )

            pretrain_model = MLPNet(X_source.shape[1], y_source.shape[1], [512, 256], 0.3).to(self.device)
            optimizer = optim.AdamW(pretrain_model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = NaNAwareMSELoss()

            best_pretrained_state = None
            best_pretrain_loss = float('inf')
            for epoch in range(pretrain_epochs):
                loss = train_epoch(pretrain_model, global_loader, optimizer, criterion, self.device)
                if loss < best_pretrain_loss:
                    best_pretrain_loss = loss
                    best_pretrained_state = copy.deepcopy(pretrain_model.state_dict())

            print(f"    🏗️ pre-trainingcomplete: loss = {best_pretrain_loss:.4f}")

            if self.split_mode == "time":
                X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s = self._split_local_data_time(
                    X_target_scaled, y_target, years_target
                )
            else:
                X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s = self._split_local_data(X_target_scaled, y_target)
            self._display_data_splits(
                experiment_type='C' if source_mode == 'all' else 'C_dev',
                country=target_country,
                global_pretrain_size=source_samples,
                target_train_size=X_train_s.shape[0],
                target_val_size=X_val_s.shape[0],
                target_test_size=X_test_s.shape[0],
                target_total=target_samples
            )

            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s)),
                batch_size=finetune_batch_size,
                shuffle=True,
                **loader_kwargs,
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val_s)),
                batch_size=eval_batch_size,
                **loader_kwargs,
            )

            def objective(trial):
                finetune_lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
                finetune_wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
                model = MLPNet(X_source.shape[1], y_source.shape[1], [512, 256], 0.3).to(self.device)
                model.load_state_dict(best_pretrained_state)
                optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=finetune_wd)
                criterion = NaNAwareMSELoss()
                best_val_r2 = -float('inf')
                best_epoch = 0
                best_state = None
                patience_counter = 0

                effective_min_epochs = max(1, min(int(finetune_min_epochs), int(finetune_epochs)))

                for epoch in range(int(finetune_epochs)):
                    train_epoch(model, train_loader, optimizer, criterion, self.device)
                    preds_val, targets_val = evaluate_pytorch(model, val_loader, self.device)
                    current_r2 = nan_aware_r2_score(targets_val, preds_val, multioutput='uniform_average')

                    if current_r2 > best_val_r2:
                        best_val_r2 = current_r2
                        best_epoch = epoch + 1
                        best_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Report progress for pruning.
                    trial.report(best_val_r2, step=epoch)
                    if use_pruner and epoch >= int(pruner_warmup_epochs) and trial.should_prune():
                        raise optuna.TrialPruned()

                    if (epoch + 1) >= effective_min_epochs and patience_counter >= int(finetune_patience):
                        break

                trial.set_user_attr('best_epoch', int(best_epoch))
                if best_state is not None:
                    trial.set_user_attr('best_val_r2', float(best_val_r2))
                return float(best_val_r2)

            pruner = optuna.pruners.MedianPruner(n_warmup_steps=int(pruner_warmup_epochs)) if use_pruner else optuna.pruners.NopPruner()
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
            study.optimize(objective, n_trials=int(n_finetune_trials), show_progress_bar=True)

            if int(stage2_trials) > 0 and float(study.best_value) >= float(stage2_min_val_r2):
                print(f"    🔁 stage2: extending Optuna by {int(stage2_trials)} trials (bestverifyR²={study.best_value:.4f})")
                study.optimize(objective, n_trials=int(stage2_trials), show_progress_bar=True)

            best_params = study.best_params
            print(f"    🔧 Fine-tuning optimization complete: bestverifyR² = {study.best_value:.4f}")

            best_epoch = int(study.best_trial.user_attrs.get('best_epoch', finetune_epochs))
            best_epoch = max(1, min(best_epoch, int(finetune_epochs)))

            final_model = MLPNet(X_source.shape[1], y_source.shape[1], [512, 256], 0.3).to(self.device)
            final_model.load_state_dict(best_pretrained_state)
            final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
            final_criterion = NaNAwareMSELoss()

            X_target_train_full = np.vstack([X_train_s, X_val_s])
            y_target_train_full = np.vstack([y_train_s, y_val_s])
            final_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_target_train_full), torch.from_numpy(y_target_train_full)),
                batch_size=finetune_batch_size,
                shuffle=True,
                **loader_kwargs,
            )

            # Train the final model for the number of epochs that achieved best validation R².
            for epoch in range(best_epoch):
                train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, self.device)

            # Compute training-set R² on the target-domain fine-tuning set.
            y_pred_train, y_true_train = evaluate_pytorch(final_model, final_train_loader, self.device)
            train_r2 = nan_aware_r2_score(y_true_train, y_pred_train, multioutput='uniform_average')

            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test_s)),
                batch_size=eval_batch_size,
                **loader_kwargs,
            )
            y_pred_test, y_true_test = evaluate_pytorch(final_model, test_loader, self.device)
            result = self._get_evaluation_scores(
                y_true_test, y_pred_test,
                f'MLP-{exp_code}-{target_country}', target_country,
                'Transfer_Learning' if source_mode == 'all' else 'Transfer_Learning_Developed'
            )
            result['training_samples'] = source_samples
            result['finetune_samples'] = target_samples
            result['optuna_best_validation_R2'] = study.best_value
            result['training_R2'] = train_r2
            result['source_mode'] = source_mode
            result['experiment_code'] = exp_code
            
            # Add sample-size and year metadata for downstream analysis.
            result['n_source'] = source_samples
            result['n_train'] = X_train_s.shape[0]
            result['n_val'] = X_val_s.shape[0]
            result['n_test'] = X_test_s.shape[0]
            result['n_total'] = target_samples
            result['years_min'] = int(years_target.min()) if len(years_target) > 0 else None
            result['years_max'] = int(years_target.max()) if len(years_target) > 0 else None
            result['n_years'] = len(np.unique(years_target)) if len(years_target) > 0 else 0
            result['is_developed'] = target_country in self.developed_countries if self.developed_countries else None
            
            # Add industry statistics (for industry-subset analysis).
            if sectors is not None:
                unique_sectors = np.unique(sectors[~pd.isnull(sectors)])
                sector_counts = pd.Series(sectors).value_counts()
                result['n_sectors'] = len(unique_sectors)
                result['dominant_sector'] = sector_counts.index[0] if len(sector_counts) > 0 else None
                result['dominant_sector_ratio'] = sector_counts.iloc[0] / len(sectors) if len(sector_counts) > 0 else None
            else:
                result['n_sectors'] = None
                result['dominant_sector'] = None
                result['dominant_sector_ratio'] = None

            print(f"    ✅ complete: trainR²={train_r2:.4f}, verifyR²={study.best_value:.4f}, testingR²={result['R2_Average']:.4f}")
            return result

        except Exception as e:
            print(f"    ❌ failure: {str(e)}")
            return {
                'model': f'MLP-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Transfer_Learning' if source_mode == 'all' else 'Transfer_Learning_Developed',
                'experiment_code': exp_code
            }

    def run_experiment_D(
        self,
        target_country,
        source_mode='all',
        pretrain_epochs=60,
        finetune_epochs=40,
        pretrain_lr=1e-3,
        finetune_lr=5e-4,
        lora_rank=8,
        lora_alpha=16,
        dropout=0.1,
        exp_code_override: str | None = None,
        pretrain_batch_size=512,
        finetune_batch_size=256,
    ):
        """experimentD/D_dev: FT-Transformer + LoRA transfer learning。"""
        exp_code = exp_code_override or ('D_dev' if source_mode == 'dev' else 'D')
        exp_name = 'developedcountry' if source_mode == 'dev' else 'global'
        print(f"  ⚡ experiment{exp_code}: {exp_name}→{target_country} FT-Transformer LoRA")

        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Transfer_Learning_FTT' if source_mode == 'all' else 'Transfer_Learning_FTT_Developed',
                'experiment_code': exp_code
            }

        X_target, y_target, years_target, target_samples, _, sectors = target_result
        X_source, y_source, source_samples = self._prepare_source_data(target_country, source_mode)

        if source_samples < 200 or target_samples < 20:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': 'Transfer_Learning_FTT' if source_mode == 'all' else 'Transfer_Learning_FTT_Developed',
                'experiment_code': exp_code
            }

        print(f"    📊 pre-trainingsamples: {source_samples:,}, fine-tuning samples: {target_samples:,}")

        try:
            scaler = StandardScaler()
            X_source_scaled = scaler.fit_transform(X_source).astype(np.float32)
            X_target_scaled = scaler.transform(X_target).astype(np.float32)

            X_src_train, y_src_train, X_src_val, y_src_val, _, _ = self._split_local_data(
                X_source_scaled, y_source, test_size=0.1, val_size=0.1
            )

            loader_kwargs = self._build_loader_kwargs()
            src_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_train), torch.from_numpy(y_src_train)),
                batch_size=pretrain_batch_size,
                shuffle=True,
                **loader_kwargs,
            )
            src_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_val), torch.from_numpy(y_src_val)),
                batch_size=pretrain_batch_size,
                **loader_kwargs,
            )

            pretrain_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=dropout
            ).to(self.device)
            pre_ckpt = None
            if self._epoch_checkpoint_root:
                pre_ckpt = os.path.join(self._epoch_checkpoint_root, f"{target_country}__{exp_code}__pretrain.pt")
            train_ft_transformer(
                pretrain_model,
                src_train_loader,
                src_val_loader,
                self.device,
                epochs=pretrain_epochs,
                lr=pretrain_lr,
                patience=15,
                checkpoint_path=pre_ckpt,
                resume=True,
            )
            pretrained_state = copy.deepcopy(pretrain_model.state_dict())

            if self.split_mode == "time":
                X_t_train, y_t_train, X_t_val, y_t_val, X_t_test, y_t_test = self._split_local_data_time(
                    X_target_scaled, y_target, years_target
                )
            else:
                X_t_train, y_t_train, X_t_val, y_t_val, X_t_test, y_t_test = self._split_local_data(X_target_scaled, y_target)
            self._display_data_splits(
                experiment_type='D' if source_mode == 'all' else 'D_dev',
                country=target_country,
                global_pretrain_size=source_samples,
                target_train_size=X_t_train.shape[0],
                target_val_size=X_t_val.shape[0],
                target_test_size=X_t_test.shape[0],
                target_total=target_samples
            )

            target_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_train), torch.from_numpy(y_t_train)),
                batch_size=finetune_batch_size,
                shuffle=True,
                **loader_kwargs,
            )
            target_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_val), torch.from_numpy(y_t_val)),
                batch_size=finetune_batch_size,
                **loader_kwargs,
            )
            target_test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_test), torch.from_numpy(y_t_test)),
                batch_size=finetune_batch_size,
                **loader_kwargs,
            )

            finetune_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=dropout,
                use_lora=True,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            ).to(self.device)
            finetune_model.load_state_dict(pretrained_state, strict=False)

            for name, param in finetune_model.named_parameters():
                if 'lora' in name or name.startswith('head'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            finetune_ckpt = None
            if self._epoch_checkpoint_root:
                finetune_ckpt = os.path.join(self._epoch_checkpoint_root, f"{target_country}__{exp_code}__finetune.pt")
            train_ft_transformer(
                finetune_model,
                target_train_loader,
                target_val_loader,
                self.device,
                epochs=finetune_epochs,
                lr=finetune_lr,
                patience=10,
                checkpoint_path=finetune_ckpt,
                resume=True,
            )

            # Compute training-set R² on the target-domain fine-tuning set.
            train_pred, train_true = evaluate_pytorch(finetune_model, target_train_loader, self.device)
            train_r2 = nan_aware_r2_score(train_true, train_pred, multioutput='uniform_average')

            # calculatevalidation setR2
            val_pred, val_true = evaluate_pytorch(finetune_model, target_val_loader, self.device)
            val_r2 = nan_aware_r2_score(val_true, val_pred, multioutput='uniform_average')

            test_pred, test_true = evaluate_pytorch(finetune_model, target_test_loader, self.device)
            result = self._get_evaluation_scores(
                test_true, test_pred,
                f'FTT-{exp_code}-{target_country}', target_country,
                'Transfer_Learning_FTT' if source_mode == 'all' else 'Transfer_Learning_FTT_Developed'
            )
            result['training_samples'] = source_samples
            result['finetune_samples'] = target_samples
            result['training_R2'] = train_r2
            result['validation_R2'] = val_r2
            result['source_mode'] = source_mode
            result['experiment_code'] = exp_code
            
            # Add sample-size and year metadata for downstream analysis.
            result['n_source'] = source_samples
            result['n_train'] = X_t_train.shape[0]
            result['n_val'] = X_t_val.shape[0]
            result['n_test'] = X_t_test.shape[0]
            result['n_total'] = target_samples
            result['years_min'] = int(years_target.min()) if len(years_target) > 0 else None
            result['years_max'] = int(years_target.max()) if len(years_target) > 0 else None
            result['n_years'] = len(np.unique(years_target)) if len(years_target) > 0 else 0
            result['is_developed'] = target_country in self.developed_countries if self.developed_countries else None
            
            # Add industry statistics (for industry-subset analysis).
            if sectors is not None:
                unique_sectors = np.unique(sectors[~pd.isnull(sectors)])
                sector_counts = pd.Series(sectors).value_counts()
                result['n_sectors'] = len(unique_sectors)
                result['dominant_sector'] = sector_counts.index[0] if len(sector_counts) > 0 else None
                result['dominant_sector_ratio'] = sector_counts.iloc[0] / len(sectors) if len(sector_counts) > 0 else None
            else:
                result['n_sectors'] = None
                result['dominant_sector'] = None
                result['dominant_sector_ratio'] = None

            print(f"    ✅ complete: trainR²={train_r2:.4f}, verifyR²={val_r2:.4f}, testingR²={result['R2_Average']:.4f}")
            return result

        except Exception as e:
            print(f"    ❌ failure: {str(e)}")
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Transfer_Learning_FTT' if source_mode == 'all' else 'Transfer_Learning_FTT_Developed',
                'experiment_code': exp_code
            }

    def run_experiment_D_full(
        self,
        target_country,
        source_mode='all',
        pretrain_epochs=60,
        finetune_epochs=40,
        pretrain_lr=1e-3,
        finetune_lr=5e-4,
        dropout=0.1,
        pretrain_batch_size=512,
        finetune_batch_size=256,
    ):
        """experimentD_full/D_full_dev: FT-Transformer Full Fine-tuning transfer learning。"""
        exp_code = 'D_full_dev' if source_mode == 'dev' else 'D_full'
        exp_name = 'developedcountry' if source_mode == 'dev' else 'global'
        print(f"  ⚡ experiment{exp_code}: {exp_name}→{target_country} FT-Transformer Full FT")

        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Transfer_Learning_FTT_Full' if source_mode == 'all' else 'Transfer_Learning_FTT_Full_Developed',
                'experiment_code': exp_code
            }

        X_target, y_target, years_target, target_samples, _, sectors = target_result
        X_source, y_source, source_samples = self._prepare_source_data(target_country, source_mode)

        if source_samples < 200 or target_samples < 20:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': 'Transfer_Learning_FTT_Full' if source_mode == 'all' else 'Transfer_Learning_FTT_Full_Developed',
                'experiment_code': exp_code
            }

        print(f"    📊 pre-trainingsamples: {source_samples:,}, fine-tuning samples: {target_samples:,}")

        try:
            scaler = StandardScaler()
            X_source_scaled = scaler.fit_transform(X_source).astype(np.float32)
            X_target_scaled = scaler.transform(X_target).astype(np.float32)

            X_src_train, y_src_train, X_src_val, y_src_val, _, _ = self._split_local_data(
                X_source_scaled, y_source, test_size=0.1, val_size=0.1
            )

            loader_kwargs = self._build_loader_kwargs()
            src_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_train), torch.from_numpy(y_src_train)),
                batch_size=pretrain_batch_size,
                shuffle=True,
                **loader_kwargs,
            )
            src_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_val), torch.from_numpy(y_src_val)),
                batch_size=pretrain_batch_size,
                **loader_kwargs,
            )

            pretrain_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=dropout
            ).to(self.device)
            pre_ckpt = None
            if self._epoch_checkpoint_root:
                pre_ckpt = os.path.join(self._epoch_checkpoint_root, f"{target_country}__{exp_code}__pretrain.pt")
            train_ft_transformer(
                pretrain_model,
                src_train_loader,
                src_val_loader,
                self.device,
                epochs=pretrain_epochs,
                lr=pretrain_lr,
                patience=15,
                checkpoint_path=pre_ckpt,
                resume=True,
            )
            pretrained_state = copy.deepcopy(pretrain_model.state_dict())

            if self.split_mode == "time":
                X_t_train, y_t_train, X_t_val, y_t_val, X_t_test, y_t_test = self._split_local_data_time(
                    X_target_scaled, y_target, years_target
                )
            else:
                X_t_train, y_t_train, X_t_val, y_t_val, X_t_test, y_t_test = self._split_local_data(X_target_scaled, y_target)
            self._display_data_splits(
                experiment_type='D_full' if source_mode == 'all' else 'D_full_dev',
                country=target_country,
                global_pretrain_size=source_samples,
                target_train_size=X_t_train.shape[0],
                target_val_size=X_t_val.shape[0],
                target_test_size=X_t_test.shape[0],
                target_total=target_samples
            )

            target_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_train), torch.from_numpy(y_t_train)),
                batch_size=finetune_batch_size,
                shuffle=True,
                **loader_kwargs,
            )
            target_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_val), torch.from_numpy(y_t_val)),
                batch_size=finetune_batch_size,
                **loader_kwargs,
            )
            target_test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_test), torch.from_numpy(y_t_test)),
                batch_size=finetune_batch_size,
                **loader_kwargs,
            )

            finetune_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=dropout
            ).to(self.device)
            finetune_model.load_state_dict(pretrained_state, strict=False)

            # Full FT: allparametercantrain
            for _, param in finetune_model.named_parameters():
                param.requires_grad = True

            finetune_ckpt = None
            if self._epoch_checkpoint_root:
                finetune_ckpt = os.path.join(self._epoch_checkpoint_root, f"{target_country}__{exp_code}__finetune.pt")
            train_ft_transformer(
                finetune_model,
                target_train_loader,
                target_val_loader,
                self.device,
                epochs=finetune_epochs,
                lr=finetune_lr,
                patience=10,
                checkpoint_path=finetune_ckpt,
                resume=True,
            )

            train_pred, train_true = evaluate_pytorch(finetune_model, target_train_loader, self.device)
            train_r2 = nan_aware_r2_score(train_true, train_pred, multioutput='uniform_average')

            val_pred, val_true = evaluate_pytorch(finetune_model, target_val_loader, self.device)
            val_r2 = nan_aware_r2_score(val_true, val_pred, multioutput='uniform_average')

            test_pred, test_true = evaluate_pytorch(finetune_model, target_test_loader, self.device)
            result = self._get_evaluation_scores(
                test_true, test_pred,
                f'FTT-{exp_code}-{target_country}', target_country,
                'Transfer_Learning_FTT_Full' if source_mode == 'all' else 'Transfer_Learning_FTT_Full_Developed'
            )
            result['training_samples'] = source_samples
            result['finetune_samples'] = target_samples
            result['training_R2'] = train_r2
            result['validation_R2'] = val_r2
            result['source_mode'] = source_mode
            result['experiment_code'] = exp_code
            
            # Add sample-size and year metadata for downstream analysis.
            result['n_source'] = source_samples
            result['n_train'] = X_t_train.shape[0]
            result['n_val'] = X_t_val.shape[0]
            result['n_test'] = X_t_test.shape[0]
            result['n_total'] = target_samples
            result['years_min'] = int(years_target.min()) if len(years_target) > 0 else None
            result['years_max'] = int(years_target.max()) if len(years_target) > 0 else None
            result['n_years'] = len(np.unique(years_target)) if len(years_target) > 0 else 0
            result['is_developed'] = target_country in self.developed_countries if self.developed_countries else None
            
            # Add industry statistics (for industry-subset analysis).
            if sectors is not None:
                unique_sectors = np.unique(sectors[~pd.isnull(sectors)])
                sector_counts = pd.Series(sectors).value_counts()
                result['n_sectors'] = len(unique_sectors)
                result['dominant_sector'] = sector_counts.index[0] if len(sector_counts) > 0 else None
                result['dominant_sector_ratio'] = sector_counts.iloc[0] / len(sectors) if len(sector_counts) > 0 else None
            else:
                result['n_sectors'] = None
                result['dominant_sector'] = None
                result['dominant_sector_ratio'] = None

            print(f"    ✅ complete: trainR²={train_r2:.4f}, verifyR²={val_r2:.4f}, testingR²={result['R2_Average']:.4f}")
            return result

        except Exception as e:
            print(f"    ❌ failure: {str(e)}")
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Transfer_Learning_FTT_Full' if source_mode == 'all' else 'Transfer_Learning_FTT_Full_Developed',
                'experiment_code': exp_code
            }

    def run_single_experiment(self, country, experiment_type, 
                               n_trials_lgbm=10, tabnet_epochs=50, tabnet_patience=10,
                               ft_epochs=80, ft_lr=1e-3, ft_dropout=0.1, **kwargs):
        """
        Run a single experiment for a country.
        
        This method dispatches to the appropriate experiment method (A, B, B_dev, C, etc.)
        based on the experiment_type parameter.
        
        Args:
            country (str): Target country code (e.g., 'CHN', 'USA')
            experiment_type (str): Experiment type ('A', 'B', 'B_dev', 'C', 'C_dev', 'D', 'D_dev', 'D_full', 'D_full_dev')
            n_trials_lgbm (int): Number of LightGBM hyperparameter trials
            tabnet_epochs (int): TabNet training epochs
            tabnet_patience (int): TabNet early stopping patience
            ft_epochs (int): FT-Transformer training epochs
            ft_lr (float): FT-Transformer learning rate
            ft_dropout (float): FT-Transformer dropout
            **kwargs: Additional experiment-specific parameters
            
        Returns:
            dict: Experiment result dictionary
        """
        exp_type = str(experiment_type).upper()
        
        if exp_type == 'A':
            return self.run_experiment_A(
                country=country,
                n_trials=n_trials_lgbm,
                tabnet_epochs=tabnet_epochs,
                tabnet_patience=tabnet_patience,
                ft_epochs=ft_epochs,
                ft_lr=ft_lr,
                ft_dropout=ft_dropout,
                **kwargs
            )
        elif exp_type == 'B':
            return self.run_experiment_B(
                target_country=country,
                n_trials=n_trials_lgbm,
                source_mode='all',
                **kwargs
            )
        elif exp_type == 'B_DEV':
            return self.run_experiment_B(
                target_country=country,
                n_trials=n_trials_lgbm,
                source_mode='dev',
                **kwargs
            )
        elif exp_type == 'C':
            return self.run_experiment_C(
                target_country=country,
                source_mode='all',
                **kwargs
            )
        elif exp_type == 'C_DEV':
            return self.run_experiment_C(
                target_country=country,
                source_mode='dev',
                **kwargs
            )
        elif exp_type == 'D':
            return self.run_experiment_D(
                target_country=country,
                source_mode='all',
                **kwargs
            )
        elif exp_type == 'D_DEV':
            return self.run_experiment_D(
                target_country=country,
                source_mode='dev',
                **kwargs
            )
        elif exp_type == 'D_FULL':
            return self.run_experiment_D_full(
                target_country=country,
                source_mode='all',
                **kwargs
            )
        elif exp_type == 'D_FULL_DEV':
            return self.run_experiment_D_full(
                target_country=country,
                source_mode='dev',
                **kwargs
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}. "
                           f"Supported: A, B, B_dev, C, C_dev, D, D_dev, D_full, D_full_dev")

# --- Convenience wrapper to run global experiments (supports a single country) ---
def run_global_experiments(df, features, targets, countries=None, min_samples=50,
                          experiments=None, runner=None, quick_run: bool = False, **kwargs):
    """
    Convenient function to run global multi-country experiments
    
    Supports two running modes：
    1. Single country: countries='CHN' or countries=['CHN']
    2. Multiple countries: countries=['CHN', 'USA', 'JPN'] or countries=None (all eligible countries meeting constraints)
    
    Args:
        df (pd.DataFrame): full dataset; must contain a 'loc' column
        features (list): List of feature column names
        targets (list): List of target variable column names
        countries (str, list, or None): needtestingcountry
            - str: a single country code, e.g., 'CHN'
            - list: a list of country codes, e.g., ['CHN', 'USA']
            - None: all eligible countries meeting the min_samples constraint
        min_samples (int): minimum sample threshold; countries below this threshold will be skipped
        experiments (list): needrunexperimentlist，optional['A', 'B', 'B_dev', 'C', 'C_dev', 'D', 'D_dev', 'D_full', 'D_full_dev']
        runner (GlobalExperimentRunner, optional): a pre-initialized runner to avoid repeated initialization
        **kwargs: other experiment parameters
            - random_state (int): random seed, default 42
            - developed_countries (list[str]): list of developed countries, used by *_dev experiments
            - n_trials_lgbm (int): number of LGBM hyperparameter optimization trials, default 20
            - tabnet_epochs/tabnet_patience/ft_epochs/ft_lr/ft_dropout: experimentAconfiguration
            - n_finetune_trials/pretrain_epochs/finetune_epochs: experimentCconfiguration
            - ft_pretrain_epochs/ft_finetune_epochs/ft_pretrain_lr/ft_finetune_lr/lora_rank/lora_alpha: experimentDconfiguration
            - ft_pretrain_epochs/ft_finetune_epochs/ft_pretrain_lr/ft_finetune_lr: experimentD_fullconfiguration
        
    Returns:
        list: a list of experiment result dictionaries
        
    Examples:
        # Run all experiments for a single country
        results = run_global_experiments(df, features, targets, countries='CHN')
        
        # Run specified experiments for specified countries
        results = run_global_experiments(df, features, targets, 
                                       countries=['CHN', 'USA', 'JPN'], 
                                       experiments=['A', 'B'])
        
        # Run all experiments for all countries meeting the constraints
        results = run_global_experiments(df, features, targets, min_samples=100)
    """
    # Create a global experiment runner.
    if experiments is None:
        experiments = ['A', 'B', 'B_dev', 'C', 'C_dev', 'D', 'D_dev', 'D_full', 'D_full_dev']
    else:
        experiments = list(dict.fromkeys(experiments))

    if runner is None:
        runner = GlobalExperimentRunner(
            df,
            features,
            targets,
            kwargs.get('random_state', 42),
            developed_countries=kwargs.get('developed_countries')
        )

    # Allow caller to override DataLoader settings for PyTorch workloads.
    if 'dataloader_num_workers' in kwargs:
        runner.dataloader_num_workers = kwargs.get('dataloader_num_workers')
    if 'dataloader_pin_memory' in kwargs:
        runner.dataloader_pin_memory = kwargs.get('dataloader_pin_memory')
    if 'dataloader_persistent_workers' in kwargs:
        runner.dataloader_persistent_workers = kwargs.get('dataloader_persistent_workers')
    if 'dataloader_prefetch_factor' in kwargs:
        runner.dataloader_prefetch_factor = kwargs.get('dataloader_prefetch_factor')
    # QUICK_RUN: override expensive defaults for a fast smoke/short run
    if quick_run:
        print("🔧 QUICK_RUN enabled: applying fast defaults for heavy experiments")
        kwargs.setdefault('n_trials_lgbm', 1)
        kwargs.setdefault('optuna_timeout_seconds', 300)
        kwargs.setdefault('lgbm_n_estimators_min', 100)
        kwargs.setdefault('lgbm_n_estimators_max', 300)
        kwargs.setdefault('xgb_n_estimators_min', 100)
        kwargs.setdefault('xgb_n_estimators_max', 300)
        kwargs.setdefault('catboost_iterations_min', 100)
        kwargs.setdefault('catboost_iterations_max', 300)
        kwargs.setdefault('lgbm_n_estimators', 200)
        kwargs.setdefault('lgbm_final_estimators', 300)
        # Experiment A deep candidates are extremely slow on CPU; skip them by default in QUICK_RUN.
        # (Tree models still run, so A* is produced quickly.)
        kwargs.setdefault('tabnet_epochs', 0)
        kwargs.setdefault('tabnet_patience', 5)
        kwargs.setdefault('ft_epochs', 0)
        kwargs.setdefault('n_finetune_trials', 1)
        kwargs.setdefault('pretrain_epochs', 5)
        kwargs.setdefault('finetune_epochs', 5)
        kwargs.setdefault('ft_pretrain_epochs', 5)
        kwargs.setdefault('ft_finetune_epochs', 5)
        kwargs.setdefault('lora_rank', 4)
        kwargs.setdefault('lora_alpha', 8)
    
    # --- Parse countries parameter and determine the evaluation set ---
    if countries is None:                       # if not specifiedcountry
        # Collect all eligible countries meeting the minimum-sample constraint.
        eligible_countries = []
        for country in runner.all_countries:
            if runner.country_quality[country]['usable'] >= min_samples:
                eligible_countries.append(country)
        print(f"Will run experiments for all qualifying countries: {len(eligible_countries)}")
        
    elif isinstance(countries, str):            # single-country string
        # convertaslistformat
        eligible_countries = [countries] if countries in runner.all_countries else []
        if not eligible_countries:
            print(f"❌ incorrect: country '{countries}' not in data")
            return []
        print(f"Will run experiments for the specified country: {countries}")
        
    elif isinstance(countries, list):           # ifyescountrylist
        # Keep only the countries that exist in the dataset.
        eligible_countries = [c for c in countries if c in runner.all_countries]
        if len(eligible_countries) != len(countries):
            missing = [c for c in countries if c not in runner.all_countries]
            print(f"⚠️ warning: withundercountrynot in data: {missing}")
        print(f"🎯 will process specified {len(eligible_countries)} countryrunexperiment")
        
    else:                                       # parametertypeincorrect
        raise ValueError("countries must be a string, a list, or None.")
    
    # Check whether there are any eligible countries.
    if not eligible_countries:
        print("❌ no qualifyingcountries meeting conditionscanwithrunexperiment")
        return []
    
    # Further filter by sample size (for user-specified countries).
    if countries is not None:                   # for specified countries, enforce the min_samples constraint
        valid_countries = []
        for country in eligible_countries:
            if runner.country_quality[country]['usable'] >= min_samples:
                valid_countries.append(country)
            else:
                print(f"⚠️ skip {country}: insufficient samples ({runner.country_quality[country]['usable']} < {min_samples})")
        eligible_countries = valid_countries
    
    if not eligible_countries:
        print("❌ No specified countries meet the minimum-sample constraint; nothing to run.")
        return []
    
    # Display the experiment plan.
    print("Experiment plan:")
    print(f"   - target country: {eligible_countries}")
    print(f"   - Experiment type: {experiments}")
    print(f"   - min_samples: {min_samples}")
    
    # --- executeexperimentloop ---
    all_results = []                            # store all experiment results

    # --- Checkpointing (persist "country × experiment" results; supports resume) ---
    checkpoint_dir = kwargs.pop('checkpoint_dir', None)
    resume = kwargs.pop('resume', True)
    checkpoint_failed = kwargs.pop('checkpoint_failed', False)
    checkpoint_run_id = kwargs.pop('checkpoint_run_id', None)

    checkpoint_root = None
    signature = None

    def _stable_json_dumps(obj) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)

    def _compute_signature() -> str:
        payload = {
            'data_shape': tuple(df.shape) if df is not None else None,
            'data_columns': list(df.columns) if hasattr(df, 'columns') else None,
            'loc_nunique': int(df['loc'].nunique()) if isinstance(df, pd.DataFrame) and 'loc' in df.columns else None,
            'min_samples': min_samples,
            'experiments': experiments,
            'features': list(features) if features is not None else None,
            'targets': list(targets) if targets is not None else None,
            'kwargs': kwargs,
        }
        digest = hashlib.sha1(_stable_json_dumps(payload).encode('utf-8')).hexdigest()
        return digest[:12]

    def _safe_name(text: str) -> str:
        # only keepfilesafe characters for name
        keep = []
        for ch in str(text):
            if ch.isalnum() or ch in ('-', '_', '.', '+'):
                keep.append(ch)
            else:
                keep.append('_')
        return ''.join(keep)

    def _atomic_pickle_dump(obj, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.pkl', dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(obj, f)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _atomic_text_write(text: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.txt', dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _checkpoint_path(country_code: str, exp_code: str) -> str:
        if checkpoint_root is None:
            raise RuntimeError('checkpoint_root is not initialized')
        filename = f"{_safe_name(country_code)}__{_safe_name(exp_code)}.pkl"
        return os.path.join(checkpoint_root, filename)

    def _maybe_load_checkpoint(country_code: str, exp_code: str):
        if checkpoint_root is None or not resume:
            return None
        path = _checkpoint_path(country_code, exp_code)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                wrapper = pickle.load(f)
            if not isinstance(wrapper, dict):
                return None
            if wrapper.get('signature') != signature:
                return None
            result = wrapper.get('result')
            if not isinstance(result, dict):
                return None
            r2_val = result.get('R2_Average', np.nan)
            try:
                r2_is_nan = bool(np.isnan(r2_val))
            except Exception:
                r2_is_nan = True
            is_failed = ('status' in result) or r2_is_nan
            if is_failed and not checkpoint_failed:
                return None
            print(f"    ⏭️ checkpoint resume: {country_code} - {exp_code} (usecheckpoint)")
            return result
        except Exception as e:
            print(f"    ⚠️ Failed to read checkpoint; re-running: {country_code}-{exp_code} ({e})")
            return None

    def _save_checkpoint(country_code: str, exp_code: str, result: dict) -> None:
        if checkpoint_root is None:
            return
        path = _checkpoint_path(country_code, exp_code)
        wrapper = {
            'signature': signature,
            'country': country_code,
            'experiment_code': exp_code,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'result': result,
        }
        try:
            _atomic_pickle_dump(wrapper, path)
        except Exception as e:
            print(f"    ⚠️ Failed to write checkpoint (continuing): {country_code}-{exp_code} ({e})")

    # Initialize checkpoint directory (stored under results/checkpoints/<signature>/).
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.getcwd(), 'results', 'checkpoints')
    signature = checkpoint_run_id or _compute_signature()
    checkpoint_root = os.path.join(checkpoint_dir, signature)
    os.makedirs(checkpoint_root, exist_ok=True)
    print(f"Checkpoint enabled: {checkpoint_root}")
    print(f"   - resume={resume}, checkpoint_failed={checkpoint_failed}")

    # Epoch-level resume checkpoint root directory (used by training loops).
    runner._epoch_checkpoint_root = os.path.join(checkpoint_root, "_epoch")
    os.makedirs(runner._epoch_checkpoint_root, exist_ok=True)

    # Write one-time run info for subsequent summaries and traceability.
    runinfo_path = os.path.join(checkpoint_root, '_RUNINFO.json')
    if not os.path.exists(runinfo_path):
        try:
            runinfo = {
                'signature': signature,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'min_samples': min_samples,
                'experiments': experiments,
                'features_count': len(features) if features is not None else None,
                'targets': list(targets) if targets is not None else None,
                'data_shape': tuple(df.shape) if df is not None else None,
            }
            _atomic_text_write(_stable_json_dumps(runinfo), runinfo_path)
        except Exception as e:
            print(f"   ⚠️ Failed to write run info (can be ignored): {e}")
    
    def _get_rss_mb():
        if psutil is None:
            return None
        try:
            return psutil.Process().memory_info().rss / (1024 ** 2)
        except Exception:
            return None

    def _get_cuda_mem_mb():
        if not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.memory_allocated(0) / (1024 ** 2)
        except Exception:
            return None

    def _get_cuda_peak_mb():
        if not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        except Exception:
            return None

    def _format_float(value, digits=4):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "nan"
        return f"{value:.{digits}f}"

    def _log_experiment_metrics(result):
        train_r2 = result.get('training_R2', np.nan)
        val_r2 = result.get('validation_R2', result.get('optuna_best_validation_R2', np.nan))
        test_r2 = result.get('R2_Average', np.nan)

        print(
            f"    📈 metrics: trainR²={_format_float(train_r2)}, verifyR²={_format_float(val_r2)}, testingR²={_format_float(test_r2)}"
        )

        run_seconds = result.get('run_seconds', None)
        mem_before = result.get('mem_rss_mb_before', None)
        mem_after = result.get('mem_rss_mb_after', None)
        mem_delta = None
        if mem_before is not None and mem_after is not None:
            mem_delta = mem_after - mem_before

        cuda_before = result.get('cuda_mem_mb_before', None)
        cuda_after = result.get('cuda_mem_mb_after', None)
        cuda_peak = result.get('cuda_mem_mb_peak', None)

        cost_parts = []
        if run_seconds is not None:
            cost_parts.append(f"time={run_seconds:.2f}s")
        if mem_before is not None and mem_after is not None:
            cost_parts.append(f"CPU mem={mem_before:.1f}→{mem_after:.1f}MB (Δ{mem_delta:.1f})")
        if cuda_before is not None or cuda_after is not None or cuda_peak is not None:
            cost_parts.append(
                f"GPU mem={_format_float(cuda_before, 1)}→{_format_float(cuda_after, 1)}MB, peak={_format_float(cuda_peak, 1)}MB"
            )
        if cost_parts:
            print("    cost: " + ", ".join(cost_parts))

    for i, country in enumerate(tqdm(eligible_countries, desc="globalexperimentprogress"), 1):
        print(f"\n[{i:2d}/{len(eligible_countries)}] country: {country}")
        print("=" * 60)
        
        # Display country-level data-quality information.
        country_info = runner.country_quality[country]
        print(
            f"country data: total samples={country_info['total']:,}, "
            f"usable labeled samples={country_info['usable']:,} ({country_info['usable_rate']:.1f}%)"
        )
        
        country_results = []                    # Store all experiment results for this country
        
        # Run experiments according to the requested experiment list.
        if 'A' in experiments:
            cached = _maybe_load_checkpoint(country, 'A')
            if cached is not None:
                candidate_rows = cached.pop('candidate_results', [])
                if isinstance(candidate_rows, list) and candidate_rows:
                    country_results.extend(candidate_rows)
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_A = runner.run_experiment_A(
                    country,
                    kwargs.get('n_trials_lgbm', 10),
                    kwargs.get('tabnet_epochs', 200),
                    kwargs.get('tabnet_patience', 30),
                    kwargs.get('ft_epochs', 80),
                    kwargs.get('ft_lr', 1e-3),
                    kwargs.get('ft_dropout', 0.1),
                    kwargs.get('lgbm_n_estimators_min', 200),
                    kwargs.get('lgbm_n_estimators_max', 1000),
                    kwargs.get('xgb_n_estimators_min', 200),
                    kwargs.get('xgb_n_estimators_max', 1000),
                    kwargs.get('catboost_iterations_min', 200),
                    kwargs.get('catboost_iterations_max', 1000),
                    kwargs.get('optuna_timeout_seconds', None),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_A['run_seconds'] = t1 - t0
                result_A['mem_rss_mb_before'] = mem_before
                result_A['mem_rss_mb_after'] = mem_after
                result_A['cuda_mem_mb_before'] = cuda_before
                result_A['cuda_mem_mb_after'] = cuda_after
                result_A['cuda_mem_mb_peak'] = cuda_peak
                candidate_rows = result_A.pop('candidate_results', [])
                if isinstance(candidate_rows, list) and candidate_rows:
                    country_results.extend(candidate_rows)
                country_results.append(result_A)
                _log_experiment_metrics(result_A)
                _save_checkpoint(country, 'A', result_A)

        if 'B' in experiments:
            cached = _maybe_load_checkpoint(country, 'B')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_B = runner.run_experiment_B(
                    country,
                    kwargs.get('n_trials_lgbm', 10),
                    source_mode='all',
                    lgbm_n_estimators=kwargs.get('lgbm_n_estimators', 1200),
                    lgbm_final_estimators=kwargs.get('lgbm_final_estimators', 2000),
                    optuna_timeout_seconds=kwargs.get('optuna_timeout_seconds', None),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_B['run_seconds'] = t1 - t0
                result_B['mem_rss_mb_before'] = mem_before
                result_B['mem_rss_mb_after'] = mem_after
                result_B['cuda_mem_mb_before'] = cuda_before
                result_B['cuda_mem_mb_after'] = cuda_after
                result_B['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_B)
                _log_experiment_metrics(result_B)
                _save_checkpoint(country, 'B', result_B)

        if 'B_dev' in experiments:
            cached = _maybe_load_checkpoint(country, 'B_dev')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_B_dev = runner.run_experiment_B(
                    country,
                    kwargs.get('n_trials_lgbm', 10),
                    source_mode='dev',
                    lgbm_n_estimators=kwargs.get('lgbm_n_estimators', 1200),
                    lgbm_final_estimators=kwargs.get('lgbm_final_estimators', 2000),
                    optuna_timeout_seconds=kwargs.get('optuna_timeout_seconds', None),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_B_dev['run_seconds'] = t1 - t0
                result_B_dev['mem_rss_mb_before'] = mem_before
                result_B_dev['mem_rss_mb_after'] = mem_after
                result_B_dev['cuda_mem_mb_before'] = cuda_before
                result_B_dev['cuda_mem_mb_after'] = cuda_after
                result_B_dev['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_B_dev)
                _log_experiment_metrics(result_B_dev)
                _save_checkpoint(country, 'B_dev', result_B_dev)

        if 'C' in experiments:
            cached = _maybe_load_checkpoint(country, 'C')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_C = runner.run_experiment_C(
                    country,
                    source_mode='all',
                    n_finetune_trials=kwargs.get('n_finetune_trials', 25),
                    pretrain_epochs=kwargs.get('pretrain_epochs', 30),
                    finetune_epochs=kwargs.get('finetune_epochs', 50),
                    pretrain_batch_size=kwargs.get('mlp_pretrain_batch_size', 1024),
                    finetune_batch_size=kwargs.get('mlp_finetune_batch_size', 64),
                    eval_batch_size=kwargs.get('mlp_eval_batch_size', 128),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_C['run_seconds'] = t1 - t0
                result_C['mem_rss_mb_before'] = mem_before
                result_C['mem_rss_mb_after'] = mem_after
                result_C['cuda_mem_mb_before'] = cuda_before
                result_C['cuda_mem_mb_after'] = cuda_after
                result_C['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_C)
                _log_experiment_metrics(result_C)
                _save_checkpoint(country, 'C', result_C)

        if 'C_dev' in experiments:
            cached = _maybe_load_checkpoint(country, 'C_dev')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_C_dev = runner.run_experiment_C(
                    country,
                    source_mode='dev',
                    n_finetune_trials=kwargs.get('n_finetune_trials', 25),
                    pretrain_epochs=kwargs.get('pretrain_epochs', 30),
                    finetune_epochs=kwargs.get('finetune_epochs', 50),
                    pretrain_batch_size=kwargs.get('mlp_pretrain_batch_size', 1024),
                    finetune_batch_size=kwargs.get('mlp_finetune_batch_size', 64),
                    eval_batch_size=kwargs.get('mlp_eval_batch_size', 128),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_C_dev['run_seconds'] = t1 - t0
                result_C_dev['mem_rss_mb_before'] = mem_before
                result_C_dev['mem_rss_mb_after'] = mem_after
                result_C_dev['cuda_mem_mb_before'] = cuda_before
                result_C_dev['cuda_mem_mb_after'] = cuda_after
                result_C_dev['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_C_dev)
                _log_experiment_metrics(result_C_dev)
                _save_checkpoint(country, 'C_dev', result_C_dev)

        if 'D' in experiments:
            cached = _maybe_load_checkpoint(country, 'D')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_D = runner.run_experiment_D(
                    country,
                    source_mode='all',
                    pretrain_epochs=kwargs.get('ft_pretrain_epochs', 60),
                    finetune_epochs=kwargs.get('ft_finetune_epochs', 40),
                    pretrain_lr=kwargs.get('ft_pretrain_lr', 1e-3),
                    finetune_lr=kwargs.get('ft_finetune_lr', 5e-4),
                    lora_rank=kwargs.get('lora_rank', 8),
                    lora_alpha=kwargs.get('lora_alpha', 16),
                    dropout=kwargs.get('ft_dropout', 0.1),
                    pretrain_batch_size=kwargs.get('ft_pretrain_batch_size', 512),
                    finetune_batch_size=kwargs.get('ft_finetune_batch_size', 256),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_D['run_seconds'] = t1 - t0
                result_D['mem_rss_mb_before'] = mem_before
                result_D['mem_rss_mb_after'] = mem_after
                result_D['cuda_mem_mb_before'] = cuda_before
                result_D['cuda_mem_mb_after'] = cuda_after
                result_D['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_D)
                _log_experiment_metrics(result_D)
                _save_checkpoint(country, 'D', result_D)

        if 'D_dev' in experiments:
            cached = _maybe_load_checkpoint(country, 'D_dev')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_D_dev = runner.run_experiment_D(
                    country,
                    source_mode='dev',
                    pretrain_epochs=kwargs.get('ft_pretrain_epochs', 60),
                    finetune_epochs=kwargs.get('ft_finetune_epochs', 40),
                    pretrain_lr=kwargs.get('ft_pretrain_lr', 1e-3),
                    finetune_lr=kwargs.get('ft_finetune_lr', 5e-4),
                    lora_rank=kwargs.get('lora_rank', 8),
                    lora_alpha=kwargs.get('lora_alpha', 16),
                    dropout=kwargs.get('ft_dropout', 0.1),
                    pretrain_batch_size=kwargs.get('ft_pretrain_batch_size', 512),
                    finetune_batch_size=kwargs.get('ft_finetune_batch_size', 256),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_D_dev['run_seconds'] = t1 - t0
                result_D_dev['mem_rss_mb_before'] = mem_before
                result_D_dev['mem_rss_mb_after'] = mem_after
                result_D_dev['cuda_mem_mb_before'] = cuda_before
                result_D_dev['cuda_mem_mb_after'] = cuda_after
                result_D_dev['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_D_dev)
                _log_experiment_metrics(result_D_dev)
                _save_checkpoint(country, 'D_dev', result_D_dev)

        if 'D_lora' in experiments:
            cached = _maybe_load_checkpoint(country, 'D_lora')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_D_lora = runner.run_experiment_D(
                    country,
                    source_mode='all',
                    pretrain_epochs=kwargs.get('ft_pretrain_epochs', 60),
                    finetune_epochs=kwargs.get('ft_finetune_epochs', 40),
                    pretrain_lr=kwargs.get('ft_pretrain_lr', 1e-3),
                    finetune_lr=kwargs.get('ft_finetune_lr', 5e-4),
                    lora_rank=kwargs.get('lora_rank', 8),
                    lora_alpha=kwargs.get('lora_alpha', 16),
                    dropout=kwargs.get('ft_dropout', 0.1),
                    pretrain_batch_size=kwargs.get('ft_pretrain_batch_size', 512),
                    finetune_batch_size=kwargs.get('ft_finetune_batch_size', 256),
                    exp_code_override='D_lora'
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_D_lora['run_seconds'] = t1 - t0
                result_D_lora['mem_rss_mb_before'] = mem_before
                result_D_lora['mem_rss_mb_after'] = mem_after
                result_D_lora['cuda_mem_mb_before'] = cuda_before
                result_D_lora['cuda_mem_mb_after'] = cuda_after
                result_D_lora['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_D_lora)
                _log_experiment_metrics(result_D_lora)
                _save_checkpoint(country, 'D_lora', result_D_lora)

        if 'D_lora_dev' in experiments:
            cached = _maybe_load_checkpoint(country, 'D_lora_dev')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_D_lora_dev = runner.run_experiment_D(
                    country,
                    source_mode='dev',
                    pretrain_epochs=kwargs.get('ft_pretrain_epochs', 60),
                    finetune_epochs=kwargs.get('ft_finetune_epochs', 40),
                    pretrain_lr=kwargs.get('ft_pretrain_lr', 1e-3),
                    finetune_lr=kwargs.get('ft_finetune_lr', 5e-4),
                    lora_rank=kwargs.get('lora_rank', 8),
                    lora_alpha=kwargs.get('lora_alpha', 16),
                    dropout=kwargs.get('ft_dropout', 0.1),
                    pretrain_batch_size=kwargs.get('ft_pretrain_batch_size', 512),
                    finetune_batch_size=kwargs.get('ft_finetune_batch_size', 256),
                    exp_code_override='D_lora_dev'
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_D_lora_dev['run_seconds'] = t1 - t0
                result_D_lora_dev['mem_rss_mb_before'] = mem_before
                result_D_lora_dev['mem_rss_mb_after'] = mem_after
                result_D_lora_dev['cuda_mem_mb_before'] = cuda_before
                result_D_lora_dev['cuda_mem_mb_after'] = cuda_after
                result_D_lora_dev['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_D_lora_dev)
                _log_experiment_metrics(result_D_lora_dev)
                _save_checkpoint(country, 'D_lora_dev', result_D_lora_dev)

        if 'D_full' in experiments:
            cached = _maybe_load_checkpoint(country, 'D_full')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_D_full = runner.run_experiment_D_full(
                    country,
                    source_mode='all',
                    pretrain_epochs=kwargs.get('ft_pretrain_epochs', 60),
                    finetune_epochs=kwargs.get('ft_finetune_epochs', 40),
                    pretrain_lr=kwargs.get('ft_pretrain_lr', 1e-3),
                    finetune_lr=kwargs.get('ft_finetune_lr', 5e-4),
                    dropout=kwargs.get('ft_dropout', 0.1),
                    pretrain_batch_size=kwargs.get('ft_pretrain_batch_size', 512),
                    finetune_batch_size=kwargs.get('ft_finetune_batch_size', 256),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_D_full['run_seconds'] = t1 - t0
                result_D_full['mem_rss_mb_before'] = mem_before
                result_D_full['mem_rss_mb_after'] = mem_after
                result_D_full['cuda_mem_mb_before'] = cuda_before
                result_D_full['cuda_mem_mb_after'] = cuda_after
                result_D_full['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_D_full)
                _log_experiment_metrics(result_D_full)
                _save_checkpoint(country, 'D_full', result_D_full)

        if 'D_full_dev' in experiments:
            cached = _maybe_load_checkpoint(country, 'D_full_dev')
            if cached is not None:
                country_results.append(cached)
                _log_experiment_metrics(cached)
            else:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mem_before = _get_rss_mb()
                cuda_before = _get_cuda_mem_mb()
                t0 = time.time()
                result_D_full_dev = runner.run_experiment_D_full(
                    country,
                    source_mode='dev',
                    pretrain_epochs=kwargs.get('ft_pretrain_epochs', 60),
                    finetune_epochs=kwargs.get('ft_finetune_epochs', 40),
                    pretrain_lr=kwargs.get('ft_pretrain_lr', 1e-3),
                    finetune_lr=kwargs.get('ft_finetune_lr', 5e-4),
                    dropout=kwargs.get('ft_dropout', 0.1),
                    pretrain_batch_size=kwargs.get('ft_pretrain_batch_size', 512),
                    finetune_batch_size=kwargs.get('ft_finetune_batch_size', 256),
                )
                t1 = time.time()
                mem_after = _get_rss_mb()
                cuda_after = _get_cuda_mem_mb()
                cuda_peak = _get_cuda_peak_mb()
                result_D_full_dev['run_seconds'] = t1 - t0
                result_D_full_dev['mem_rss_mb_before'] = mem_before
                result_D_full_dev['mem_rss_mb_after'] = mem_after
                result_D_full_dev['cuda_mem_mb_before'] = cuda_before
                result_D_full_dev['cuda_mem_mb_after'] = cuda_after
                result_D_full_dev['cuda_mem_mb_peak'] = cuda_peak
                country_results.append(result_D_full_dev)
                _log_experiment_metrics(result_D_full_dev)
                _save_checkpoint(country, 'D_full_dev', result_D_full_dev)
        
        # Add this country's results to the global results.
        all_results.extend(country_results)
        
        # --- Display experiment summary for this country ---
        successful = [r for r in country_results if 'status' not in r and not np.isnan(r.get('R2_Average', np.nan))]
        failed = [r for r in country_results if 'status' in r or np.isnan(r.get('R2_Average', np.nan))]
        
        if successful:                          # ifhassuccessexperiment
            avg_r2 = np.mean([r['R2_Average'] for r in successful])
            print(f"  📊 {country} complete: averageR² = {avg_r2:.4f} ({len(successful)}/{len(country_results)}experiments succeeded)")
            
            # Display per-experiment detailed results.
            for result in successful:
                exp_type = result['experiment_type']
                model_name = result['model']
                r2_score = result['R2_Average']
                print(f"     - {model_name}: R² = {r2_score:.4f}")
        else:                                   # if notsuccessexperiment
            print(f"  ❌ {country} nonesuccessexperiment")
        
        # displayfailureexperiment（ifhas）
        if failed:
            print(f"  ⚠️ failureexperiment:")
            for result in failed:
                model_name = result.get('model', 'Unknown')
                error_msg = result.get('status', result.get('error_message', 'Unknown error'))
                print(f"     - {model_name}: {error_msg}")
    
    # --- Global experiment summary ---
    print(f"\n{'='*80}")
    print(f"🎉 globalexperimentcomplete!")
    
    # statisticssuccessandfailureexperimentcount
    total_experiments = len(all_results)
    successful_experiments = len([r for r in all_results if 'status' not in r and not np.isnan(r.get('R2_Average', np.nan))])
    failed_experiments = total_experiments - successful_experiments
    
    print(f"📊 experimentstatistics:")
    print(f"   - total experiments: {total_experiments}")
    print(f"   - successexperiment: {successful_experiments}")
    print(f"   - failureexperiment: {failed_experiments}")
    print(
        f"   - success rate: {(successful_experiments/total_experiments)*100:.1f}%"
        if total_experiments > 0
        else "   - success rate: 0%"
    )
    
    # Summarize results by experiment type.
    if successful_experiments > 0:
        print(f"\n📈 successexperimentresultssummary:")
        for exp_code in experiments:
            exp_results = [
                r for r in all_results
                if r.get('experiment_code') == exp_code and not np.isnan(r.get('R2_Average', np.nan))
            ]
            if exp_results:
                avg_r2 = np.mean([r['R2_Average'] for r in exp_results])
                print(f"   - experiment{exp_code}: {len(exp_results)}successful, averageR² = {avg_r2:.4f}")
    
    return all_results                          # returnallexperimentresults