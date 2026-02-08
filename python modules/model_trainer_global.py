# ==============================================================================
# Global Country Loop Experiment Runner - Supports Experiments A, B, C and Extended Experiments
# for Single and Batch Country Processing
# ==============================================================================
import os                                     # OS interface for file path operations
import pandas as pd                           # Data processing and analysis library
import numpy as np                            # Numerical computing library, provides efficient array operations
import lightgbm as lgb                        # LightGBM gradient boosting framework, suitable for tabular data
from sklearn.multioutput import MultiOutputRegressor  # Scikit-learn multi-output regression wrapper
from sklearn.model_selection import train_test_split  # Dataset splitting utility
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Evaluation metrics
from sklearn.preprocessing import StandardScaler # Feature standardization utility
import torch                                  # PyTorch deep learning framework
import torch.nn as nn                         # PyTorch neural network module
import torch.optim as optim                   # PyTorch optimizer module
from torch.utils.data import TensorDataset, DataLoader  # PyTorch data loading utilities
import optuna                                 # Automatic hyperparameter optimization library
from tqdm.auto import tqdm                    # Progress bar display library
import copy                                   # Python deep copy module
import warnings

# --- Global Settings ---
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Set Optuna log level to WARNING, reduce output noise
warnings.filterwarnings('ignore')                      # Suppress warning messages for clean output

# --- NaN-aware Loss Functions and Evaluation Functions ---
import torch.nn.functional as F

class NaNAwareMSELoss(nn.Module):
    """
    MSE loss function that supports NaN values in targets
    
    This loss function intelligently handles NaN values in target variables,
    computing loss only on valid values. Perfectly suited for carbon emission
    data where some target variables may be missing.
    """
    def __init__(self):
        super(NaNAwareMSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        """
        Forward pass to compute loss
        
        Args:
            predictions: Model predictions, shape (batch_size, n_targets)
            targets: True target values, shape (batch_size, n_targets), may contain NaN
            
        Returns:
            MSE loss computed only on valid values
        """
        # Create valid value mask: True indicates non-NaN values
        valid_mask = ~torch.isnan(targets)
        
        # If all values in current batch are NaN, return zero loss
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        # Compute MSE loss only on valid values
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        return F.mse_loss(valid_predictions, valid_targets)

def nan_aware_r2_score(y_true, y_pred, multioutput='uniform_average'):
    """
    R² score calculation function that supports NaN values
    
    Args:
        y_true: True values, may contain NaN
        y_pred: Predicted values
        multioutput: Multi-output handling method
        
    Returns:
        R² score with intelligent NaN handling
    """
    if y_true.ndim == 1:
        # Single target case
        valid_mask = ~np.isnan(y_true)
        if valid_mask.sum() < 2:  # Need at least 2 valid samples to compute R²
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

# --- PyTorch MLP Model Definition (Original Architecture) ---
class MLPNet(nn.Module):
    """
    Multi-Layer Perceptron Network class, designed for multi-target regression on tabular data
    
    Architecture features:
    - Uses LayerNorm instead of BatchNorm, more suitable for small batches and tabular data
    - Supports multi-target output (simultaneously predicting multiple carbon emission metrics)
    - Flexible hidden layer architecture configuration
    """
    
    def __init__(self, n_features, n_targets, hidden_layers, dropout_rate):
        """
        Initialize MLP network structure
        
        Args:
            n_features (int): Number of input features
            n_targets (int): Number of output targets (e.g., 6 different carbon emission metrics)
            hidden_layers (list): List of hidden layer neuron counts, e.g., [512, 256]
            dropout_rate (float): Dropout regularization rate to prevent overfitting
        """
        super(MLPNet, self).__init__()          # Call parent class nn.Module initialization
        layers = []                             # Create empty list to store network layers
        input_dim = n_features                  # Set first layer input dimension to feature count
        
        # Build hidden layer sequence
        for hidden_dim in hidden_layers:        # Iterate over each hidden layer's neuron count
            layers.append(nn.Linear(input_dim, hidden_dim))    # Add linear transformation layer: y = xW^T + b
            layers.append(nn.LayerNorm(hidden_dim))            # Add layer normalization: normalize all features per sample
            layers.append(nn.ReLU())                           # Add ReLU activation: max(0, x), introduces non-linearity
            layers.append(nn.Dropout(dropout_rate))            # Add Dropout layer: randomly zeros some neurons during training
            input_dim = hidden_dim              # Update next layer's input dimension
        
        layers.append(nn.Linear(input_dim, n_targets))         # Add output layer: direct linear transform to target dimension
        self.model = nn.Sequential(*layers)                    # Combine all layers into a sequential container

    def forward(self, x):
        """
        Forward pass function, defines data flow through the network
        
        Args:
            x (torch.Tensor): Input feature tensor, shape (batch_size, n_features)
            
        Returns:
            torch.Tensor: Prediction result tensor, shape (batch_size, n_targets)
        """
        return self.model(x)                    # Pass input data through entire network

# --- PyTorch Training and Evaluation Helper Functions (Modified - NaN Support) ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Execute one complete training epoch - Supports NaN-aware training
    
    Args:
        model (nn.Module): PyTorch model to train
        dataloader (DataLoader): Batch loader for training data
        optimizer (optim.Optimizer): Optimizer responsible for updating model parameters based on gradients
        criterion (nn.Module): Loss function (should be NaN-aware)
        device (torch.device): Computing device ('cpu' or 'cuda:0' etc.)
        
    Returns:
        float: Average loss for this epoch
    """
    model.train()                              # Set model to training mode
    total_loss = 0                             # Initialize cumulative loss
    valid_batches = 0                          # Track valid batch count
    
    # Iterate over each batch in dataloader
    for X_batch, y_batch in dataloader:
        # Move input data and targets to specified computing device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Check if current batch has valid target values
        if torch.isnan(y_batch).all():
            continue  # Skip batch if all values are NaN
        
        optimizer.zero_grad()                  # Zero out previous gradients
        outputs = model(X_batch)               # Forward pass: compute predictions through model
        loss = criterion(outputs, y_batch)     # Compute loss (NaN-aware)
        
        # Check if loss is valid
        if torch.isnan(loss) or loss.item() == 0:
            continue  # Skip invalid loss
            
        loss.backward()                        # Backward pass: compute gradients
        optimizer.step()                       # Parameter update: adjust model weights based on gradients
        total_loss += loss.item()              # Accumulate loss value
        valid_batches += 1                     # Increment valid batch count
    
    return total_loss / valid_batches if valid_batches > 0 else 0.0

def evaluate_pytorch(model, dataloader, device):
    """
    Evaluate PyTorch model performance on given dataset
    
    Args:
        model (nn.Module): Trained PyTorch model
        dataloader (DataLoader): Batch loader for evaluation data
        device (torch.device): Computing device
        
    Returns:
        tuple: (predictions matrix, true values matrix), both as numpy arrays
    """
    model.eval()                               # Set model to evaluation mode (disable dropout, fix batch norm params)
    all_preds, all_targets = [], []            # Initialize lists to store all batch predictions and true values
    
    with torch.no_grad():                      # Disable gradient computation (save memory and computation time)
        for X_batch, y_batch in dataloader:   # Iterate over each evaluation batch
            X_batch = X_batch.to(device)       # Move input data to computing device
            preds = model(X_batch)             # Get predictions through model
            all_preds.append(preds.cpu().numpy())     # Move predictions from GPU to CPU and convert to numpy
            all_targets.append(y_batch.numpy())       # Convert true values to numpy
    
    return np.vstack(all_preds), np.vstack(all_targets)  # Vertically stack all batch results


def nan_aware_r2_score(y_true, y_pred, multioutput='uniform_average'):
    """
    R² score calculation function that supports NaN values
    (Reference: model_trainer_multiMLP_ynan.py)
    
    Args:
        y_true: True values, may contain NaN
        y_pred: Predicted values
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


# --- Developed Countries List Definition ---
DEVELOPED_COUNTRIES = [
    'USA', 'CAN', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'CHE', 'AUT', 'IRL', 'LUX', 'PRT',
    'SWE', 'NOR', 'DNK', 'FIN', 'GRC', 'POL', 'JPN', 'KOR', 'TWN', 'HKG', 'SGP',
    'ISR', 'SAU', 'ARE', 'KWT', 'QAT', 'AUS', 'NZL', 'CHL', 'BMU', 'CYM'
]

# --- FT-Transformer Model Definition ---
class FTTransformerModel(nn.Module):
    """
    FT-Transformer (Feature Tokenizer + Transformer) Model
    
    Transformer architecture designed specifically for tabular data, supports LoRA fine-tuning.
    """
    def __init__(self, n_features, n_targets, d_model=128, n_heads=4, n_layers=3, 
                 dropout=0.1, use_lora=False, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.d_model = d_model
        self.use_lora = use_lora
        
        # Feature tokenizer: map each feature to d_model dimensional embedding
        self.feature_embeddings = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets)
        )
        
        # LoRA adaptation layers (optional)
        if use_lora:
            self.lora_A = nn.Linear(d_model, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, d_model, bias=False)
            self.lora_alpha = lora_alpha
            self.lora_rank = lora_rank
            nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Feature tokenization: (batch, n_features) -> (batch, n_features, d_model)
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.feature_embeddings(x)  # (batch, n_features, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_features+1, d_model)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # LoRA adaptation
        if self.use_lora:
            lora_out = self.lora_B(self.lora_A(x))
            x = x + (self.lora_alpha / self.lora_rank) * lora_out
        
        # Use CLS token for prediction
        cls_output = x[:, 0, :]
        return self.head(cls_output)


def train_ft_transformer(model, train_loader, val_loader, device, epochs=50, 
                        lr=1e-3, patience=10, checkpoint_path=None, resume=False):
    """
    Train FT-Transformer model
    
    Args:
        model: FTTransformerModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computing device
        epochs: Training epochs
        lr: Learning rate
        patience: Early stopping patience
        checkpoint_path: Checkpoint save path
        resume: Whether to resume from checkpoint
        
    Returns:
        best_val_r2: Best validation R²
    """
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    criterion = NaNAwareMSELoss()
    
    best_val_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        print(f"      Resumed from checkpoint: epoch {start_epoch}, best_val_r2={best_val_r2:.4f}")
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0
        valid_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                valid_batches += 1
        
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        val_r2 = nan_aware_r2_score(targets, preds, multioutput='uniform_average')
        
        # Early stopping check
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_r2': best_val_r2
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_r2


# --- Redesigned Global Loop Experiment Runner ---
class GlobalExperimentRunner:
    """
    Global Country Loop Experiment Runner - Supports Single and Batch Country Processing
    
    Supports multiple experiment modes:
    - Experiment A: Country Self-Modeling - Train and test using single-country data
    - Experiment B: Global Baseline - Train on global data, test on single country
    - Experiment B_dev: Developed Countries Baseline - Train on developed country data, test on single country
    - Experiment C: Transfer Learning - Global pre-training + single-country fine-tuning
    - Experiment C_dev: Developed Countries Transfer Learning - Developed pre-training + single-country fine-tuning
    - Experiment D_lora: FT-Transformer + LoRA Transfer Learning - Global pre-training + LoRA fine-tuning
    - Experiment D_lora_dev: FT-Transformer + LoRA Transfer Learning - Developed pre-training + LoRA fine-tuning
    - Experiment D_full: FT-Transformer Full Fine-tuning - Global pre-training + full parameter fine-tuning
    - Experiment D_full_dev: FT-Transformer Full Fine-tuning - Developed pre-training + full parameter fine-tuning
    
    Core Design Principles:
    - Unified data interface and evaluation standards
    - Flexible source and target domain configuration
    - NaN-aware data processing
    - Supports single country or batch country processing
    """
    
    def __init__(self, full_data, features, targets, random_state=42, developed_countries=None):
        """
        Initialize Global Experiment Runner
        
        Args:
            full_data (pd.DataFrame): Complete DataFrame with all country data, must have 'loc' column
            features (list): List of feature column names
            targets (list): List of target variable column names
            random_state (int): Random seed for reproducibility
            developed_countries (list, optional): List of developed countries for B_dev/C_dev experiments
        """
        self.full_data = full_data.copy()       # Deep copy original data to avoid accidental modification
        self.features = features                # Store feature column names
        self.targets = targets                  # Store target variable column names
        self.random_state = random_state        # Store random seed
        
        # Developed countries list: prioritize passed list, otherwise use default
        if developed_countries is not None:
            self.developed_countries = [c for c in developed_countries if c in full_data['loc'].values]
        else:
            self.developed_countries = [c for c in DEVELOPED_COUNTRIES if c in full_data['loc'].values]
        
        # Device detection: prioritize GPU for acceleration, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"🌍 Global Experiment Runner initialized. Computing device: {device_info}")
        
        # Get all countries sorted for consistency
        self.all_countries = sorted(self.full_data['loc'].unique())
        print(f"📊 Found {len(self.all_countries)} countries/regions")
        print(f"📊 Developed countries count: {len(self.developed_countries)}")
        
        # Data quality pre-analysis for subsequent country filtering
        self._analyze_global_data_quality()
    
    def _analyze_global_data_quality(self):
        """
        Analyze global data quality, generate data quality report for each country
        
        This method:
        1. Computes total sample count per country
        2. Computes usable sample count (both target and features non-null)
        3. Computes data completeness rate
        4. Sorts by sample count and displays top 10
        """
        print("🔍 Analyzing global data quality...")
        
        country_quality = []  # Store data quality info for each country
        
        for country in self.all_countries:     # Iterate over all countries
            country_data = self.full_data[self.full_data['loc'] == country]
            total_samples = len(country_data)
            
            # Compute usable samples
            target_valid = country_data[self.targets].notnull().any(axis=1).sum()      # At least one target non-null
            feature_complete = country_data[self.features].notnull().all(axis=1).sum() # All features non-null
            usable_samples = (
                country_data[self.targets].notnull().any(axis=1) & 
                country_data[self.features].notnull().all(axis=1)
            ).sum()
            
            country_quality.append({
                'country': country,
                'total': total_samples,
                'usable': usable_samples,
                'usable_rate': (usable_samples / total_samples * 100) if total_samples > 0 else 0
            })
        
        # Sort by usable samples (descending)
        country_quality.sort(key=lambda x: x['usable'], reverse=True)
        
        # Display top 10 countries by sample count
        print("📈 Top 10 Countries by Sample Count:")
        for i, cq in enumerate(country_quality[:10], 1):
            print(f"  {i:2d}. {cq['country']}: {cq['usable']:,} usable samples ({cq['usable_rate']:.1f}%)")
        
        # Store as dictionary for easy lookup
        self.country_quality = {cq['country']: cq for cq in country_quality}
    
    def _display_data_splits(self, experiment_type, country, **split_info):
        """
        Display experiment dataset split information with detailed data source info
        
        Args:
            experiment_type (str): Experiment type ('A', 'B', 'C')
            country (str): Target country code
            **split_info: Keyword arguments for dataset split information
        """
        exp_names = {
            'A': 'Country Self-Modeling',
            'B': 'Global Baseline', 
            'C': 'Transfer Learning'
        }
        
        print(f"    📊 Experiment {experiment_type} ({exp_names.get(experiment_type, 'Unknown')}) - {country} Dataset Split and Sources:")
        
        if experiment_type == 'A':
            # Experiment A: Single country data split
            train_size = split_info.get('train_size', 0)
            test_size = split_info.get('test_size', 0)
            val_size = split_info.get('val_size', 0)
            total_usable = split_info.get('total_usable', 0)
            
            print(f"      📍 Data Source Strategy: Use only target country {country} local data")
            print(f"      📈 {country} Total Usable Samples: {total_usable:,}")
            print(f"      🏋️ Training Set: {train_size:,} samples ({train_size/total_usable*100:.1f}%) - Source: {country} country data")
            if val_size > 0:
                print(f"      ✅ Validation Set: {val_size:,} samples ({val_size/total_usable*100:.1f}%) - Source: {country} country data")
            print(f"      🎯 Test Set: {test_size:,} samples ({test_size/total_usable*100:.1f}%) - Source: {country} country data")
            print(f"      💡 Note: All datasets from target country, testing in-domain generalization")
            
        elif experiment_type == 'B':
            # Experiment B: Global training, single country testing
            global_train_size = split_info.get('global_train_size', 0)
            global_val_size = split_info.get('global_val_size', 0)
            target_test_size = split_info.get('target_test_size', 0)
            global_total = split_info.get('global_total', 0)
            
            print(f"      📍 Data Source Strategy: Global data training → Target country testing")
            print(f"      🌍 Global Training Data (excluding {country}):")
            print(f"        🏋️ Training Set: {global_train_size:,} samples ({global_train_size/global_total*100:.1f}%) - Source: Multi-country global data (excluding {country})")
            print(f"        ✅ Validation Set: {global_val_size:,} samples ({global_val_size/global_total*100:.1f}%) - Source: Multi-country global data (excluding {country})")
            print(f"      🎯 Target Test Data:")
            print(f"        🎯 Test Set: {target_test_size:,} samples - Source: {country} country data")
            print(f"      💡 Note: Train on global data, test cross-domain generalization on target country")
            
        elif experiment_type == 'C':
            # Experiment C: Transfer Learning - Global pre-training + single country fine-tuning
            global_pretrain_size = split_info.get('global_pretrain_size', 0)
            target_train_size = split_info.get('target_train_size', 0)
            target_val_size = split_info.get('target_val_size', 0)
            target_test_size = split_info.get('target_test_size', 0)
            target_total = split_info.get('target_total', 0)
            
            print(f"      📍 Data Source Strategy: Global pre-training → Target country fine-tuning")
            print(f"      🌍 Phase 1 - Global Pre-training:")
            print(f"        🏗️ Pre-training Set: {global_pretrain_size:,} samples - Source: Multi-country global data (excluding {country})")
            print(f"      🎯 Phase 2 - {country} Fine-tuning and Testing:")
            print(f"        📈 Total Usable Samples: {target_total:,} - Source: {country} country data")
            print(f"        🏋️ Fine-tuning Training Set: {target_train_size:,} samples ({target_train_size/target_total*100:.1f}%) - Source: {country} country data")
            if target_val_size > 0:
                print(f"        ✅ Fine-tuning Validation Set: {target_val_size:,} samples ({target_val_size/target_total*100:.1f}%) - Source: {country} country data")
            print(f"        🎯 Test Set: {target_test_size:,} samples ({target_test_size/target_total*100:.1f}%) - Source: {country} country data")
            print(f"      💡 Note: Pre-train universal representations on global data, then fine-tune to adapt to local features")
    
    def _prepare_country_data(self, country):
        """
        Prepare cleaned modeling data for specified country (no-fill strategy)
        
        Args:
            country (str): Country code (e.g., 'CHN', 'USA')
            
        Returns:
            tuple or None: (X, y, sample_count, original_count) if sufficient data, otherwise None
                - X (np.ndarray): Feature matrix, shape (n_samples, n_features)
                - y (np.ndarray): Target matrix, shape (n_samples, n_targets), no NaN
                - sample_count (int): Usable sample count
                - original_count (int): Original sample count (before cleaning)
        """
        # Filter data for specified country
        country_data = self.full_data[self.full_data['loc'] == country].copy()
        original_count = len(country_data)
        
        # Check if data exists
        if original_count == 0:
            return None
        
        # No-fill strategy: keep only samples where all targets are valid and features are complete
        all_targets_valid_mask = country_data[self.targets].notnull().all(axis=1)
        feature_complete_mask = country_data[self.features].notnull().all(axis=1)
        usable_mask = all_targets_valid_mask & feature_complete_mask
        
        usable_count = usable_mask.sum()
        if usable_count < 10:  # Minimum sample check (need at least 10 for modeling)
            return None
        
        # Extract cleaned data
        clean_data = country_data[usable_mask]
        X = clean_data[self.features].values.astype(np.float32)
        y = clean_data[self.targets].values.astype(np.float32)
        
        # Verify: ensure no NaN (core of no-fill strategy)
        assert not np.isnan(X).any(), "Feature matrix should not contain NaN"
        assert not np.isnan(y).any(), "Target matrix should not contain NaN (no-fill strategy)"
        
        return X, y, usable_count, original_count
    
    def _prepare_global_data(self, exclude_country=None):
        """
        Prepare global data (optionally excluding a country) - No-fill strategy
        
        Args:
            exclude_country (str, optional): Country code to exclude, for Experiments B and C
            
        Returns:
            tuple: (X_global, y_global, sample_count)
                - X_global (np.ndarray): Global feature matrix
                - y_global (np.ndarray): Global target matrix, no NaN
                - sample_count (int): Global usable sample count
        """
        # Filter data based on whether to exclude specific country
        if exclude_country:
            global_data = self.full_data[self.full_data['loc'] != exclude_country].copy()
        else:
            global_data = self.full_data.copy()
        
        # No-fill strategy: keep only samples where all targets are valid and features are complete
        all_targets_valid_mask = global_data[self.targets].notnull().all(axis=1)
        feature_complete_mask = global_data[self.features].notnull().all(axis=1)
        usable_mask = all_targets_valid_mask & feature_complete_mask
        
        usable_count = usable_mask.sum()
        if usable_count < 100:  # Global data needs more samples
            print(f"    ⚠️ Insufficient global usable samples: {usable_count} < 100")
            return None, None, 0
        
        # Extract cleaned data
        clean_global_data = global_data[usable_mask]
        X_global = clean_global_data[self.features].values.astype(np.float32)
        y_global = clean_global_data[self.targets].values.astype(np.float32)
        
        # Verify: ensure no NaN (core of no-fill strategy)
        assert not np.isnan(X_global).any(), "Global feature matrix should not contain NaN"
        assert not np.isnan(y_global).any(), "Global target matrix should not contain NaN (no-fill strategy)"
        
        return X_global, y_global, usable_count
    
    def _prepare_developed_data(self, exclude_country=None):
        """
        Prepare developed countries data (optionally excluding a country) - No-fill strategy
        
        Args:
            exclude_country (str, optional): Country code to exclude
            
        Returns:
            tuple: (X_dev, y_dev, sample_count)
                - X_dev (np.ndarray): Developed countries feature matrix
                - y_dev (np.ndarray): Developed countries target matrix, no NaN
                - sample_count (int): Developed countries usable sample count
        """
        # Filter developed countries data (excluding target country)
        dev_countries = [c for c in self.developed_countries if c != exclude_country]
        dev_data = self.full_data[self.full_data['loc'].isin(dev_countries)].copy()
        
        # No-fill strategy: keep only samples where all targets are valid and features are complete
        all_targets_valid_mask = dev_data[self.targets].notnull().all(axis=1)
        feature_complete_mask = dev_data[self.features].notnull().all(axis=1)
        usable_mask = all_targets_valid_mask & feature_complete_mask
        
        usable_count = usable_mask.sum()
        if usable_count < 100:
            print(f"    ⚠️ Insufficient developed countries usable samples: {usable_count} < 100")
            return None, None, 0
        
        clean_dev_data = dev_data[usable_mask]
        X_dev = clean_dev_data[self.features].values.astype(np.float32)
        y_dev = clean_dev_data[self.targets].values.astype(np.float32)
        
        assert not np.isnan(X_dev).any(), "Developed countries feature matrix should not contain NaN"
        assert not np.isnan(y_dev).any(), "Developed countries target matrix should not contain NaN"
        
        return X_dev, y_dev, usable_count
    
    def _handle_nan_targets(self, y):
        """
        Handle NaN values in target variables - No-fill strategy version
        
        Under no-fill strategy, training data should not contain NaN because we filtered
        for all valid targets during data preparation.
        
        Args:
            y (np.ndarray): Target matrix, should not contain NaN
            
        Returns:
            np.ndarray: Validated target matrix
        """
        # Verify: under no-fill strategy, training data should not have NaN
        if np.isnan(y).any():
            raise ValueError("Under no-fill strategy, training data should not contain NaN. Please check data preparation logic.")
        
        return y.copy()  # Return copy to avoid accidental modification of original data
    
    def _get_evaluation_scores(self, y_true, y_pred, model_name, country, experiment_type):
        """
        Compute evaluation metrics, calculate R², MAE, RMSE for each target variable separately
        
        Args:
            y_true (np.ndarray): True values matrix
            y_pred (np.ndarray): Predicted values matrix
            model_name (str): Model name identifier
            country (str): Country code
            experiment_type (str): Experiment type identifier
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Initialize results dictionary with basic info
        results = {
            'model': model_name,
            'country': country,
            'experiment_type': experiment_type
        }
        
        avg_r2 = 0          # For computing average R²
        valid_targets = 0   # Track valid target count
        
        # Compute metrics for each target variable separately
        for i, target in enumerate(self.targets):
            # Find valid samples (exclude NaN)
            valid_mask = ~(np.isnan(y_true[:, i]) | np.isnan(y_pred[:, i]))
            
            # Only compute metrics if valid sample count > 1 and true values have variance
            if valid_mask.sum() > 1 and len(np.unique(y_true[valid_mask, i])) > 1:
                # Compute each metric
                r2 = r2_score(y_true[valid_mask, i], y_pred[valid_mask, i])          # R² coefficient of determination
                mae = mean_absolute_error(y_true[valid_mask, i], y_pred[valid_mask, i])  # Mean absolute error
                rmse = np.sqrt(mean_squared_error(y_true[valid_mask, i], y_pred[valid_mask, i]))  # Root mean squared error
                
                # Store each metric
                results[f'R2_{target}'] = r2
                results[f'MAE_{target}'] = mae
                results[f'RMSE_{target}'] = rmse
                
                avg_r2 += r2        # Accumulate R²
                valid_targets += 1  # Increment valid target count
            else:
                # Invalid target, set to NaN
                results[f'R2_{target}'] = np.nan
                results[f'MAE_{target}'] = np.nan
                results[f'RMSE_{target}'] = np.nan
        
        # Compute average R²
        results['R2_Average'] = avg_r2 / valid_targets if valid_targets > 0 else np.nan
        results['Valid_Targets'] = valid_targets
        
        return results
    
    def run_experiment_A(self, country, n_trials=20):
        """
        Experiment A: Country Self-Modeling
        
        Train and test using single country's own data, evaluate in-domain model performance
        
        Args:
            country (str): Target country code
            n_trials (int): LGBM hyperparameter optimization trial count
            
        Returns:
            dict: Evaluation results dictionary
        """
        print(f"  🏠 Experiment A: {country} Self-Modeling")
        
        # Prepare country data
        country_result = self._prepare_country_data(country)
        if country_result is None:
            return {
                'model': f'LGBM-A-{country}',
                'country': country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': 'Country_Self_Modeling'
            }
        
        X, y, sample_count, original_count = country_result
        print(f"    📊 Original samples: {original_count:,}, Usable samples: {sample_count:,} ({sample_count/original_count*100:.1f}%)")
        
        try:
            # Use standard data split ratio: 70%-15%-15%
            test_size = 0.15   # Test set 15%
            val_size = 0.15    # Validation set 15%, training set auto 70%
            
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            # Second split: train vs val
            val_ratio = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            # Keep original data (with NaN) for evaluation
            y_train_raw = y_train.copy()
            y_val_raw = y_val.copy()
            y_test_raw = y_test.copy()
            
            # Display dataset split info
            self._display_data_splits(
                experiment_type='A',
                country=country,
                train_size=X_train.shape[0],
                val_size=X_val.shape[0],
                test_size=X_test.shape[0],
                total_usable=sample_count,
                original_size=original_count
            )
            
            # Define Optuna optimization objective function
            def objective(trial):
                """
                LGBM hyperparameter optimization objective function
                
                Args:
                    trial: Optuna trial object
                    
                Returns:
                    float: Average R² score on validation set
                """
                # Set parameter ranges based on sample count
                max_estimators = min(500, max(50, sample_count * 10))
                max_leaves = min(50, max(10, sample_count // 2))
                max_depth = min(15, max(3, sample_count // 10))
                
                # Define hyperparameter search space
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'n_estimators': max_estimators,
                    'random_state': self.random_state,
                    'n_jobs': 1,
                    'verbose': -1,
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, max_leaves),
                    'max_depth': trial.suggest_int('max_depth', 3, max_depth),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                
                # Create and train model (using fully valid samples)
                model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
                
                # Find samples where all targets are valid for training
                valid_sample_mask = ~np.isnan(y_train_raw).any(axis=1)
                
                if valid_sample_mask.sum() < 10:
                    return -1.0
                
                # Train using fully valid samples (no filling needed)
                X_train_valid = X_train[valid_sample_mask]
                y_train_valid = y_train_raw[valid_sample_mask]
                
                # Train model (data completely NaN-free, LightGBM can process normally)
                model.fit(X_train_valid, y_train_valid)
                preds = model.predict(X_val)
                
                # Use NaN-aware R² to calculate validation performance (based on original validation data, may contain NaN)
                r2_avg = nan_aware_r2_score(y_val_raw, preds, multioutput='uniform_average')
                
                return r2_avg if not np.isnan(r2_avg) else -1.0  # Return average R², return -1 if invalid
            
            # Execute hyperparameter optimization
            study = optuna.create_study(direction='maximize')  # Maximize R²
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # Train final model with best parameters (NaN-aware single-target method)
            best_params = study.best_params
            final_params = {
                **best_params,
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': min(1000, max(100, sample_count * 15)),  # Final model uses more trees
                'random_state': self.random_state,
                'n_jobs': 1,
                'verbose': -1
            }
            
            # Train final model (smart NaN handling)
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**final_params))
            
            # Find samples with at least one valid target value
            valid_sample_mask = ~np.isnan(y_train_raw).all(axis=1)
            X_train_valid = X_train[valid_sample_mask]
            y_train_valid = y_train_raw[valid_sample_mask].copy()
            
            # Train final model (using fully valid samples)
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**final_params))
            
            # Find samples where all targets are valid for training
            valid_sample_mask = ~np.isnan(y_train_raw).any(axis=1)
            X_train_valid = X_train[valid_sample_mask]
            y_train_valid = y_train_raw[valid_sample_mask]
            
            final_model.fit(X_train_valid, y_train_valid)  # Train with fully valid samples
            
            # Predict on test set
            y_pred_test = final_model.predict(X_test)
            
            # Calculate evaluation metrics - using NaN-aware method (based on original test data)
            result = self._get_evaluation_scores(
                y_test_raw, y_pred_test, 
                f'LGBM-A-{country}', country, 'Country_Self_Modeling'
            )
            result['training_samples'] = original_count  # Original sample count
            result['valid_training_samples'] = valid_sample_mask.sum()  # Add fully valid training sample count
            result['data_efficiency'] = valid_sample_mask.sum() / original_count  # Data utilization rate (based on original sample count)
            
            print(f"    ✅ Complete: R² = {result['R2_Average']:.4f} (valid training samples: {valid_sample_mask.sum()}/{original_count}, utilization: {result['data_efficiency']:.1%})")
            return result
            
        except Exception as e:                  # Catch and handle any exceptions
            print(f"    ❌ Failed: {str(e)}")
            return {
                'model': f'LGBM-A-{country}',
                'country': country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Country_Self_Modeling'
            }
    
    def run_experiment_B(self, target_country, n_trials=25):
        """
        Experiment B: Global Baseline Modeling
        
        Train with global data excluding target country, test on target country, evaluate cross-domain generalization
        
        Args:
            target_country (str): Target test country
            n_trials (int): LGBM hyperparameter optimization trial count
            
        Returns:
            dict: Evaluation results dictionary
        """
        print(f"  🌍 Experiment B: Global→{target_country} Baseline Modeling")
        
        # Prepare target country test data
        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'LGBM-B-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Global_Baseline'
            }
        
        X_target, y_target, target_samples, target_original = target_result
        
        # Prepare global training data (excluding target country)
        X_global, y_global, global_samples = self._prepare_global_data(exclude_country=target_country)
        
        if global_samples < 100:               # Global data volume check
            return {
                'model': f'LGBM-B-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_global_data',
                'experiment_type': 'Global_Baseline'
            }
        
        print(f"    📊 Global training samples: {global_samples:,}, Target test samples: {target_samples:,}")
        
        try:
            # Preserve original NaN in target variables, no filling
            y_global_raw = y_global.copy()  # Preserve original NaN in global data
            y_target_raw = y_target.copy()  # Preserve original NaN in target country data
            
            # Standard data split: global data split 70%-15%-15%
            test_size = 0.15
            val_size = 0.15
            
            # First split: train+validation vs test
            X_temp, X_test_global, y_temp, y_test_global = train_test_split(
                X_global, y_global_raw, test_size=test_size, random_state=self.random_state
            )
            
            # Second split: train vs validation
            val_ratio = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            # Display dataset split info
            self._display_data_splits(
                experiment_type='B',
                country=target_country,
                global_train_size=X_train.shape[0],
                global_val_size=X_val.shape[0],
                target_test_size=target_samples,
                global_total=global_samples
            )
            
            # Define Optuna optimization objective function
            def objective(trial):
                """
                Global model hyperparameter optimization objective function
                
                Args:
                    trial: Optuna trial object
                    
                Returns:
                    float: R² score on validation set
                """
                params = {
                    'objective': 'regression_l1',           # Use L1 loss (more robust to outliers)
                    'metric': 'rmse',
                    'n_estimators': 1000,                   # More trees (large global data)
                    'random_state': self.random_state,
                    'n_jobs': -1,                           # Multi-threading (large global data, worth parallelizing)
                    'verbose': -1,
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),      # Relatively conservative learning rate
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),                 # More leaf nodes
                    'max_depth': trial.suggest_int('max_depth', 5, 20),                     # Deeper trees
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                
                # Create and train model (using fully valid samples)
                model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
                
                # Find samples with all valid targets for training
                valid_sample_mask = ~np.isnan(y_train).any(axis=1)
                
                if valid_sample_mask.sum() < 10:  # If too few fully valid samples
                    return -1.0
                
                # Train with fully valid samples (no filling needed)
                X_train_valid = X_train[valid_sample_mask]
                y_train_valid = y_train[valid_sample_mask]
                
                # Train model (data has no NaN, LightGBM can handle normally)
                model.fit(X_train_valid, y_train_valid)
                preds = model.predict(X_val)
                
                # Use NaN-aware R² for validation performance (based on original validation data, may contain NaN)
                r2_avg = nan_aware_r2_score(y_val, preds, multioutput='uniform_average')
                
                return r2_avg if not np.isnan(r2_avg) else -1.0
            # Execute hyperparameter optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # Train final model with best parameters (NaN-aware single-target approach)
            best_params = study.best_params
            final_params = {**best_params, 'n_estimators': 2000, 'random_state': self.random_state, 'n_jobs': -1}
            
            # Train final model (using fully valid samples)
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**final_params))
            
            # Find samples with all valid targets for training
            valid_sample_mask = ~np.isnan(y_train).any(axis=1)
            X_train_valid = X_train[valid_sample_mask]
            y_train_valid = y_train[valid_sample_mask]
            
            final_model.fit(X_train_valid, y_train_valid)  # Train with fully valid samples
            
            # Test on target country
            y_pred_target = final_model.predict(X_target)
            
            # Calculate evaluation metrics - using original data (contains NaN)
            result = self._get_evaluation_scores(
                y_target_raw, y_pred_target,
                f'LGBM-B-{target_country}', target_country, 'Global_Baseline'
            )
            result['training_samples'] = global_samples   # Global training samples
            result['test_samples'] = target_samples       # Target test samples
            
            print(f"    ✅ Completed: R² = {result['R2_Average']:.4f}")
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return {
                'model': f'LGBM-B-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Global_Baseline'
            }
    
    def run_experiment_C(self, target_country, n_finetune_trials=25, pretrain_epochs=30, finetune_epochs=50):
        """
        Experiment C: Transfer Learning Modeling
        
        Three-stage transfer learning:
        1. Pre-train MLP on global data
        2. Fine-tune on target country data
        3. Evaluate on target country test set
        
        Args:
            target_country (str): Target country
            n_finetune_trials (int): Number of fine-tuning hyperparameter optimization trials
            pretrain_epochs (int): Pre-training epochs
            finetune_epochs (int): Fine-tuning epochs
            
        Returns:
            dict: Evaluation result dictionary
        """
        print(f"  🔄 Experiment C: Global→{target_country} Transfer Learning")
        
        # Prepare target country data
        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'MLP-C-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Transfer_Learning'
            }
        
        X_target, y_target, target_samples, target_original = target_result
        
        # Prepare global pre-training data
        X_global, y_global, global_samples = self._prepare_global_data(exclude_country=target_country)
        
        # Data volume check
        if global_samples < 100 or target_samples < 20:
            return {
                'model': f'MLP-C-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': 'Transfer_Learning'
            }
        
        print(f"    📊 Pre-training samples: {global_samples:,}, Fine-tuning samples: {target_samples:,}")
        
        try:
            # --- Feature standardization ---
            # Create separate scalers for pre-training and fine-tuning
            pretrain_scaler = StandardScaler()              # Scaler for pre-training phase
            X_global_scaled = pretrain_scaler.fit_transform(X_global)  # Standardize global data
            
            finetune_scaler = StandardScaler()              # Scaler for fine-tuning phase
            X_target_scaled = finetune_scaler.fit_transform(X_target)  # Standardize target data
            
            # Keep original NaN in target variables, no filling
            y_global_raw = y_global.copy()  # Keep original NaN in global data
            y_target_raw = y_target.copy()  # Keep original NaN in target country data
            
            # --- 1. Pre-training phase ---
            print(f"    🏗️ Pre-training phase...")
            
            # Create DataLoader for global data - using original NaN data
            global_dataset = TensorDataset(torch.from_numpy(X_global_scaled), torch.from_numpy(y_global_raw))
            global_loader = DataLoader(global_dataset, batch_size=1024, shuffle=True)  # Large batch, suitable for big data
            
            # Create pre-training model
            pretrain_model = MLPNet(
                X_global.shape[1],                  # Number of input features
                y_global.shape[1],                  # Number of output targets
                [512, 256],                         # Hidden layer configuration
                0.3                                 # Dropout rate
            ).to(self.device)                       # Move to compute device
            
            # Set optimizer and loss function - using NaN-aware loss function
            optimizer = optim.AdamW(pretrain_model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = NaNAwareMSELoss()  # Use NaN-aware loss function
            
            # Pre-training loop
            best_pretrained_state = None           # Save best pre-trained model state
            best_pretrain_loss = float('inf')      # Track best loss
            
            for epoch in range(pretrain_epochs):
                loss = train_epoch(pretrain_model, global_loader, optimizer, criterion, self.device)
                if loss < best_pretrain_loss:      # If loss improves
                    best_pretrain_loss = loss
                    best_pretrained_state = copy.deepcopy(pretrain_model.state_dict())  # Save model state
            
            print(f"    🏗️ Pre-training completed: loss = {best_pretrain_loss:.4f}")
            
            # --- 2. Fine-tuning phase ---
            print(f"    🔧 Fine-tuning phase...")
            
            # Target country data split - standard ratio 70%-15%-15%
            test_size = 0.15   # Test set 15%
            val_size = 0.15    # Validation set 15%, training set automatically 70%
            
            # First split: train+validation vs test
            X_temp, X_target_test, y_temp, y_target_test = train_test_split(
                X_target_scaled, y_target_raw, test_size=test_size, random_state=self.random_state
            )
            
            # Second split: train vs validation
            val_ratio = val_size / (1.0 - test_size)
            X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            # Display data split information
            self._display_data_splits(
                experiment_type='C',
                country=target_country,
                global_pretrain_size=global_samples,
                target_train_size=X_train_s.shape[0],
                target_val_size=X_val_s.shape[0],
                target_test_size=X_target_test.shape[0],
                target_total=target_samples
            )
            
            # Create fine-tuning data loaders
            train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s)), 
                                    batch_size=64, shuffle=True)    # Small batch, suitable for small data
            val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val_s)), 
                                  batch_size=128)                  # Validation batch
            
            # Define fine-tuning Optuna objective function
            def objective(trial):
                """
                Hyperparameter optimization objective function for fine-tuning phase
                
                Args:
                    trial: Optuna trial object
                    
                Returns:
                    float: Best R² score on validation set
                """
                # Fine-tuning hyperparameter search space (relatively conservative)
                finetune_lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)           # Small learning rate
                finetune_wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True) # Weight decay
                
                # Create model and load pre-trained weights
                model = MLPNet(X_global.shape[1], y_global.shape[1], [512, 256], 0.3).to(self.device)
                model.load_state_dict(best_pretrained_state)    # Start from pre-trained
                
                # Set fine-tuning optimizer and NaN-aware loss function
                optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=finetune_wd)
                criterion = NaNAwareMSELoss()  # Use NaN-aware loss function
                
                best_val_r2 = -float('inf')         # Track best validation R² for this trial
                
                # Fine-tuning training loop
                for epoch in range(finetune_epochs):
                    train_epoch(model, train_loader, optimizer, criterion, self.device)  # Train one epoch
                    preds_val, targets_val = evaluate_pytorch(model, val_loader, self.device)  # Validation evaluation
                    current_r2 = r2_score(targets_val, preds_val, multioutput='uniform_average')  # Calculate R²
                    if current_r2 > best_val_r2:   # Update best R²
                        best_val_r2 = current_r2
                
                return best_val_r2
            
            # Execute fine-tuning hyperparameter optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_finetune_trials, show_progress_bar=False)
            best_params = study.best_params
            
            print(f"    🔧 Fine-tuning optimization completed: Best validation R² = {study.best_value:.4f}")
            
            # --- 3. Final training and evaluation ---
            # Final fine-tuning with best parameters
            final_model = MLPNet(X_global.shape[1], y_global.shape[1], [512, 256], 0.3).to(self.device)
            final_model.load_state_dict(best_pretrained_state)  # Load pre-trained weights
            
            # Set optimizer and NaN-aware loss function with best hyperparameters
            final_optimizer = optim.AdamW(final_model.parameters(), 
                                        lr=best_params['lr'], weight_decay=best_params['weight_decay'])
            final_criterion = NaNAwareMSELoss()  # Use NaN-aware loss function
            
            # Fine-tune on complete target training data
            X_target_train_full = np.vstack([X_train_s, X_val_s])  # Merge training and validation sets
            y_target_train_full = np.vstack([y_train_s, y_val_s])  # Merge training and validation sets
            
            final_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_target_train_full), torch.from_numpy(y_target_train_full)), 
                                          batch_size=64, shuffle=True)
            
            for epoch in range(finetune_epochs):   # Final fine-tuning loop
                train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, self.device)
            
            # Final evaluation on test set
            test_loader = DataLoader(TensorDataset(torch.from_numpy(X_target_test), torch.from_numpy(y_target_test)), 
                                   batch_size=128)
            y_pred_test, y_true_test = evaluate_pytorch(final_model, test_loader, self.device)
            
            # Calculate final evaluation metrics
            result = self._get_evaluation_scores(
                y_true_test, y_pred_test,
                f'MLP-C-{target_country}', target_country, 'Transfer_Learning'
            )
            result['training_samples'] = global_samples     # Pre-training samples
            result['finetune_samples'] = target_samples     # Fine-tuning samples
            
            print(f"    ✅ Completed: R² = {result['R2_Average']:.4f}")
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return {
                'model': f'MLP-C-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Transfer_Learning'
            }

    def run_experiment_B_dev(self, target_country, n_trials=25):
        """
        Experiment B_dev: Developed Countries Baseline Modeling
        
        Train on developed countries data (excluding target country), test on target country
        
        Args:
            target_country (str): Target test country
            n_trials (int): Number of LGBM hyperparameter optimization trials
            
        Returns:
            dict: Evaluation results dictionary
        """
        print(f"  🌍 Experiment B_dev: Developed Countries→{target_country} Baseline Modeling")
        
        # Prepare target country test data
        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'LGBM-B_dev-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Developed_Baseline'
            }
        
        X_target, y_target, target_samples, target_original = target_result
        
        # Prepare developed country training data
        X_dev, y_dev, dev_samples = self._prepare_developed_data(exclude_country=target_country)
        
        if dev_samples is None or dev_samples < 100:
            return {
                'model': f'LGBM-B_dev-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_developed_data',
                'experiment_type': 'Developed_Baseline'
            }
        
        print(f"    📊 Developed country training samples: {dev_samples:,}, Target country test samples: {target_samples:,}")
        
        try:
            # Split train/validation set (80%-20%)
            X_train, X_val, y_train, y_val = train_test_split(
                X_dev, y_dev, test_size=0.2, random_state=self.random_state
            )
            
            # Optuna hyperparameter optimization
            def objective(trial):
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'random_state': self.random_state,
                    'n_estimators': 1000,
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 32, 256),
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'verbosity': -1
                }
                model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return nan_aware_r2_score(y_val, preds, multioutput='uniform_average')
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # Train final model with best parameters
            best_params = study.best_params
            best_params.update({'n_estimators': 1500, 'random_state': self.random_state, 'verbosity': -1})
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**best_params))
            final_model.fit(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]))
            
            # Evaluate on target country
            y_pred_target = final_model.predict(X_target)
            result = self._get_evaluation_scores(
                y_target, y_pred_target,
                f'LGBM-B_dev-{target_country}', target_country, 'Developed_Baseline'
            )
            result['training_samples'] = dev_samples
            result['test_samples'] = target_samples
            result['optuna_best_r2'] = study.best_value
            
            print(f"    ✅ Completed: R² = {result['R2_Average']:.4f}")
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return {
                'model': f'LGBM-B_dev-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Developed_Baseline'
            }

    def run_experiment_C_dev(self, target_country, n_finetune_trials=25, pretrain_epochs=30, finetune_epochs=50):
        """
        Experiment C_dev: Developed Countries Transfer Learning Modeling
        
        Pre-train MLP on developed country data, fine-tune on target country data
        
        Args:
            target_country (str): Target country
            n_finetune_trials (int): Number of fine-tuning hyperparameter optimization trials
            pretrain_epochs (int): Pre-training epochs
            finetune_epochs (int): Fine-tuning epochs
            
        Returns:
            dict: Evaluation result dictionary
        """
        print(f"  🔄 Experiment C_dev: Developed Countries→{target_country} Transfer Learning")
        
        # Prepare target country data
        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'MLP-C_dev-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': 'Transfer_Learning_Developed'
            }
        
        X_target, y_target, target_samples, target_original = target_result
        
        # Prepare developed country pre-training data
        X_dev, y_dev, dev_samples = self._prepare_developed_data(exclude_country=target_country)
        
        if dev_samples is None or dev_samples < 100 or target_samples < 20:
            return {
                'model': f'MLP-C_dev-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': 'Transfer_Learning_Developed'
            }
        
        print(f"    📊 Developed country pre-training samples: {dev_samples:,}, Fine-tuning samples: {target_samples:,}")
        
        try:
            # Feature standardization
            pretrain_scaler = StandardScaler()
            X_dev_scaled = pretrain_scaler.fit_transform(X_dev).astype(np.float32)
            
            finetune_scaler = StandardScaler()
            X_target_scaled = finetune_scaler.fit_transform(X_target).astype(np.float32)
            
            y_dev_raw = y_dev.copy()
            y_target_raw = y_target.copy()
            
            # 1. Pre-training phase
            print(f"    🏗️ Pre-training phase...")
            dev_dataset = TensorDataset(torch.from_numpy(X_dev_scaled), torch.from_numpy(y_dev_raw))
            dev_loader = DataLoader(dev_dataset, batch_size=1024, shuffle=True)
            
            pretrain_model = MLPNet(X_dev.shape[1], y_dev.shape[1], [512, 256], 0.3).to(self.device)
            optimizer = optim.AdamW(pretrain_model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = NaNAwareMSELoss()
            
            best_pretrained_state = None
            best_pretrain_loss = float('inf')
            
            for epoch in range(pretrain_epochs):
                loss = train_epoch(pretrain_model, dev_loader, optimizer, criterion, self.device)
                if loss < best_pretrain_loss:
                    best_pretrain_loss = loss
                    best_pretrained_state = copy.deepcopy(pretrain_model.state_dict())
            
            print(f"    🏗️ Pre-training completed: loss = {best_pretrain_loss:.4f}")
            
            # 2. Fine-tuning phase
            print(f"    🔧 Fine-tuning phase...")
            
            test_size = 0.15
            val_size = 0.15
            
            X_temp, X_target_test, y_temp, y_target_test = train_test_split(
                X_target_scaled, y_target_raw, test_size=test_size, random_state=self.random_state
            )
            
            val_ratio = val_size / (1.0 - test_size)
            X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s)), 
                                    batch_size=64, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val_s)), 
                                  batch_size=128)
            
            def objective(trial):
                finetune_lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
                finetune_wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
                
                model = MLPNet(X_dev.shape[1], y_dev.shape[1], [512, 256], 0.3).to(self.device)
                model.load_state_dict(best_pretrained_state)
                
                optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=finetune_wd)
                criterion = NaNAwareMSELoss()
                
                best_val_r2 = -float('inf')
                
                for epoch in range(finetune_epochs):
                    train_epoch(model, train_loader, optimizer, criterion, self.device)
                    preds_val, targets_val = evaluate_pytorch(model, val_loader, self.device)
                    current_r2 = nan_aware_r2_score(targets_val, preds_val, multioutput='uniform_average')
                    if current_r2 > best_val_r2:
                        best_val_r2 = current_r2
                
                return best_val_r2
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_finetune_trials, show_progress_bar=False)
            best_params = study.best_params
            
            print(f"    🔧 Fine-tuning optimization completed: Best validation R² = {study.best_value:.4f}")
            
            # 3. Final training and evaluation
            final_model = MLPNet(X_dev.shape[1], y_dev.shape[1], [512, 256], 0.3).to(self.device)
            final_model.load_state_dict(best_pretrained_state)
            
            final_optimizer = optim.AdamW(final_model.parameters(), 
                                        lr=best_params['lr'], weight_decay=best_params['weight_decay'])
            final_criterion = NaNAwareMSELoss()
            
            X_target_train_full = np.vstack([X_train_s, X_val_s])
            y_target_train_full = np.vstack([y_train_s, y_val_s])
            
            final_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_target_train_full), torch.from_numpy(y_target_train_full)), 
                                          batch_size=64, shuffle=True)
            
            for epoch in range(finetune_epochs):
                train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, self.device)
            
            test_loader = DataLoader(TensorDataset(torch.from_numpy(X_target_test), torch.from_numpy(y_target_test)), 
                                   batch_size=128)
            y_pred_test, y_true_test = evaluate_pytorch(final_model, test_loader, self.device)
            
            result = self._get_evaluation_scores(
                y_true_test, y_pred_test,
                f'MLP-C_dev-{target_country}', target_country, 'Transfer_Learning_Developed'
            )
            result['training_samples'] = dev_samples
            result['finetune_samples'] = target_samples
            
            print(f"    ✅ Completed: R² = {result['R2_Average']:.4f}")
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return {
                'model': f'MLP-C_dev-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': 'Transfer_Learning_Developed'
            }

    def run_experiment_D_lora(self, target_country, source_mode='all', pretrain_epochs=60, 
                              finetune_epochs=40, lora_rank=8, lora_alpha=16):
        """
        Experiment D_lora/D_lora_dev: FT-Transformer + LoRA Transfer Learning
        
        Use FT-Transformer architecture, pre-train then use LoRA for efficient fine-tuning
        
        Args:
            target_country (str): Target country
            source_mode (str): Data source mode, 'all'=Global, 'dev'=Developed Countries
            pretrain_epochs (int): Pre-training epochs
            finetune_epochs (int): Fine-tuning epochs
            lora_rank (int): LoRA rank
            lora_alpha (int): LoRA scaling coefficient
            
        Returns:
            dict: Evaluation result dictionary
        """
        exp_code = 'D_lora_dev' if source_mode == 'dev' else 'D_lora'
        exp_name = 'Developed Countries' if source_mode == 'dev' else 'Global'
        print(f"  ⚡ Experiment {exp_code}: {exp_name}→{target_country} FT-Transformer LoRA")
        
        # Prepare target country data
        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': f'Transfer_Learning_FTT_LoRA{"_Developed" if source_mode == "dev" else ""}'
            }
        
        X_target, y_target, target_samples, target_original = target_result
        
        # Prepare source domain data
        if source_mode == 'dev':
            X_source, y_source, source_samples = self._prepare_developed_data(exclude_country=target_country)
        else:
            X_source, y_source, source_samples = self._prepare_global_data(exclude_country=target_country)
        
        if source_samples is None or source_samples < 200 or target_samples < 20:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': f'Transfer_Learning_FTT_LoRA{"_Developed" if source_mode == "dev" else ""}'
            }
        
        print(f"    📊 Pre-training samples: {source_samples:,}, Fine-tuning samples: {target_samples:,}")
        
        try:
            # Feature standardization
            scaler = StandardScaler()
            X_source_scaled = scaler.fit_transform(X_source).astype(np.float32)
            X_target_scaled = scaler.transform(X_target).astype(np.float32)
            
            # Source domain data split
            X_src_train, X_src_val, y_src_train, y_src_val = train_test_split(
                X_source_scaled, y_source, test_size=0.1, random_state=self.random_state
            )
            
            src_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_train), torch.from_numpy(y_src_train)),
                batch_size=512, shuffle=True
            )
            src_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_val), torch.from_numpy(y_src_val)),
                batch_size=512
            )
            
            # 1. Pre-train FT-Transformer
            print(f"    🏗️ FT-Transformer pre-training phase...")
            pretrain_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=0.1
            ).to(self.device)
            
            best_val_r2 = train_ft_transformer(
                pretrain_model, src_train_loader, src_val_loader, self.device,
                epochs=pretrain_epochs, lr=1e-3, patience=15
            )
            pretrained_state = copy.deepcopy(pretrain_model.state_dict())
            print(f"    🏗️ Pre-training complete: validation R² = {best_val_r2:.4f}")
            
            # Target domain data split
            test_size = 0.15
            val_size = 0.15
            
            X_temp, X_t_test, y_temp, y_t_test = train_test_split(
                X_target_scaled, y_target, test_size=test_size, random_state=self.random_state
            )
            val_ratio = val_size / (1.0 - test_size)
            X_t_train, X_t_val, y_t_train, y_t_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            target_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_train), torch.from_numpy(y_t_train)),
                batch_size=256, shuffle=True
            )
            target_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_val), torch.from_numpy(y_t_val)),
                batch_size=256
            )
            target_test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_test), torch.from_numpy(y_t_test)),
                batch_size=256
            )
            
            # 2. LoRA fine-tuning
            print(f"    🔧 LoRA fine-tuning phase...")
            finetune_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=0.1,
                use_lora=True,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            ).to(self.device)
            finetune_model.load_state_dict(pretrained_state, strict=False)
            
            # Freeze non-LoRA parameters
            for name, param in finetune_model.named_parameters():
                if 'lora' in name or name.startswith('head'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            best_finetune_r2 = train_ft_transformer(
                finetune_model, target_train_loader, target_val_loader, self.device,
                epochs=finetune_epochs, lr=5e-4, patience=10
            )
            
            # 3. Test evaluation
            finetune_model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in target_test_loader:
                    X_batch = X_batch.to(self.device)
                    preds = finetune_model(X_batch)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(y_batch.numpy())
            
            y_pred_test = np.vstack(all_preds)
            y_true_test = np.vstack(all_targets)
            
            result = self._get_evaluation_scores(
                y_true_test, y_pred_test,
                f'FTT-{exp_code}-{target_country}', target_country,
                f'Transfer_Learning_FTT_LoRA{"_Developed" if source_mode == "dev" else ""}'
            )
            result['training_samples'] = source_samples
            result['finetune_samples'] = target_samples
            result['lora_rank'] = lora_rank
            result['lora_alpha'] = lora_alpha
            result['source_mode'] = source_mode
            
            print(f"    ✅ Complete: R² = {result['R2_Average']:.4f}")
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': f'Transfer_Learning_FTT_LoRA{"_Developed" if source_mode == "dev" else ""}'
            }

    def run_experiment_D_full(self, target_country, source_mode='all', pretrain_epochs=60, finetune_epochs=40):
        """
        Experiment D_full/D_full_dev: FT-Transformer Full Parameter Fine-tuning Transfer Learning
        
        Use FT-Transformer architecture, pre-train then perform full parameter fine-tuning
        
        Args:
            target_country (str): Target country
            source_mode (str): Data source mode, 'all'=Global, 'dev'=Developed Countries
            pretrain_epochs (int): Pre-training epochs
            finetune_epochs (int): Fine-tuning epochs
            
        Returns:
            dict: Evaluation result dictionary
        """
        exp_code = 'D_full_dev' if source_mode == 'dev' else 'D_full'
        exp_name = 'Developed Countries' if source_mode == 'dev' else 'Global'
        print(f"  ⚡ Experiment {exp_code}: {exp_name}→{target_country} FT-Transformer Full Fine-tuning")
        
        # Prepare target country data
        target_result = self._prepare_country_data(target_country)
        if target_result is None:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_target_data',
                'experiment_type': f'Transfer_Learning_FTT_Full{"_Developed" if source_mode == "dev" else ""}'
            }
        
        X_target, y_target, target_samples, target_original = target_result
        
        # Prepare source domain data
        if source_mode == 'dev':
            X_source, y_source, source_samples = self._prepare_developed_data(exclude_country=target_country)
        else:
            X_source, y_source, source_samples = self._prepare_global_data(exclude_country=target_country)
        
        if source_samples is None or source_samples < 200 or target_samples < 20:
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'insufficient_data',
                'experiment_type': f'Transfer_Learning_FTT_Full{"_Developed" if source_mode == "dev" else ""}'
            }
        
        print(f"    📊 Pre-training samples: {source_samples:,}, Fine-tuning samples: {target_samples:,}")
        
        try:
            # Feature standardization
            scaler = StandardScaler()
            X_source_scaled = scaler.fit_transform(X_source).astype(np.float32)
            X_target_scaled = scaler.transform(X_target).astype(np.float32)
            
            # Source domain data split
            X_src_train, X_src_val, y_src_train, y_src_val = train_test_split(
                X_source_scaled, y_source, test_size=0.1, random_state=self.random_state
            )
            
            src_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_train), torch.from_numpy(y_src_train)),
                batch_size=512, shuffle=True
            )
            src_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_src_val), torch.from_numpy(y_src_val)),
                batch_size=512
            )
            
            # 1. Pre-train FT-Transformer
            print(f"    🏗️ FT-Transformer pre-training phase...")
            pretrain_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=0.1
            ).to(self.device)
            
            best_val_r2 = train_ft_transformer(
                pretrain_model, src_train_loader, src_val_loader, self.device,
                epochs=pretrain_epochs, lr=1e-3, patience=15
            )
            pretrained_state = copy.deepcopy(pretrain_model.state_dict())
            print(f"    🏗️ Pre-training complete: validation R² = {best_val_r2:.4f}")
            
            # Target domain data split
            test_size = 0.15
            val_size = 0.15
            
            X_temp, X_t_test, y_temp, y_t_test = train_test_split(
                X_target_scaled, y_target, test_size=test_size, random_state=self.random_state
            )
            val_ratio = val_size / (1.0 - test_size)
            X_t_train, X_t_val, y_t_train, y_t_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            target_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_train), torch.from_numpy(y_t_train)),
                batch_size=256, shuffle=True
            )
            target_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_val), torch.from_numpy(y_t_val)),
                batch_size=256
            )
            target_test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_t_test), torch.from_numpy(y_t_test)),
                batch_size=256
            )
            
            # 2. Full parameter fine-tuning
            print(f"    🔧 Full parameter fine-tuning phase...")
            finetune_model = FTTransformerModel(
                n_features=X_source.shape[1],
                n_targets=y_source.shape[1],
                dropout=0.1
            ).to(self.device)
            finetune_model.load_state_dict(pretrained_state)
            
            # All parameters trainable
            for param in finetune_model.parameters():
                param.requires_grad = True
            
            best_finetune_r2 = train_ft_transformer(
                finetune_model, target_train_loader, target_val_loader, self.device,
                epochs=finetune_epochs, lr=5e-4, patience=10
            )
            
            # 3. Test evaluation
            finetune_model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in target_test_loader:
                    X_batch = X_batch.to(self.device)
                    preds = finetune_model(X_batch)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(y_batch.numpy())
            
            y_pred_test = np.vstack(all_preds)
            y_true_test = np.vstack(all_targets)
            
            result = self._get_evaluation_scores(
                y_true_test, y_pred_test,
                f'FTT-{exp_code}-{target_country}', target_country,
                f'Transfer_Learning_FTT_Full{"_Developed" if source_mode == "dev" else ""}'
            )
            result['training_samples'] = source_samples
            result['finetune_samples'] = target_samples
            result['source_mode'] = source_mode
            
            print(f"    ✅ Complete: R² = {result['R2_Average']:.4f}")
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'model': f'FTT-{exp_code}-{target_country}',
                'country': target_country,
                'R2_Average': np.nan,
                'status': 'modeling_error',
                'error_message': str(e),
                'experiment_type': f'Transfer_Learning_FTT_Full{"_Developed" if source_mode == "dev" else ""}'
            }


# --- Convenience function to run global experiments (supports single country) ---
def run_global_experiments(df, features, targets, countries=None, min_samples=50, 
                          experiments=['A', 'B', 'C'], **kwargs):
    """
    Convenience function to run global country loop experiments
    
    Supports two run modes:
    1. Single country: countries='CHN' or countries=['CHN']
    2. Multiple countries: countries=['CHN', 'USA', 'JPN'] or countries=None (all eligible countries)
    
    Args:
        df (pd.DataFrame): Complete DataFrame with 'loc' column required
        features (list): List of feature column names
        targets (list): List of target variable column names
        countries (str, list, or None): Countries to test
            - str: Single country code, e.g., 'CHN'
            - list: List of country codes, e.g., ['CHN', 'USA']
            - None: All countries meeting min_samples threshold
        min_samples (int): Minimum sample threshold, countries below this will be skipped
        experiments (list): List of experiments to run, options: ['A', 'B', 'C']
        **kwargs: Other experiment parameters
            - random_state (int): Random seed, default 42
            - n_trials_lgbm (int): LGBM hyperparameter optimization trials, default 20
            - n_finetune_trials (int): Fine-tuning hyperparameter optimization trials, default 25
            - pretrain_epochs (int): Pre-training epochs, default 30
            - finetune_epochs (int): Fine-tuning epochs, default 50
        
    Returns:
        list: List containing all experiment results, each element is a result dictionary
        
    Examples:
        # Run all experiments for a single country
        results = run_global_experiments(df, features, targets, countries='CHN')
        
        # Run specified experiments for multiple specified countries
        results = run_global_experiments(df, features, targets, 
                                       countries=['CHN', 'USA', 'JPN'], 
                                       experiments=['A', 'B'])
        
        # Run all experiments for all eligible countries
        results = run_global_experiments(df, features, targets, min_samples=100)
    """
    # Create global experiment runner
    runner = GlobalExperimentRunner(df, features, targets, kwargs.get('random_state', 42))
    
    # --- Process countries parameter to determine list of countries to test ---
    if countries is None:                       # If no countries specified
        # Filter all countries meeting minimum sample threshold
        eligible_countries = []
        for country in runner.all_countries:
            if runner.country_quality[country]['usable'] >= min_samples:
                eligible_countries.append(country)
        print(f"🎯 Will run experiments on all {len(eligible_countries)} eligible countries")
        
    elif isinstance(countries, str):            # If single country string
        # Convert to list format
        eligible_countries = [countries] if countries in runner.all_countries else []
        if not eligible_countries:
            print(f"❌ Error: Country '{countries}' not in data")
            return []
        print(f"🎯 Will run experiments on single country {countries}")
        
    elif isinstance(countries, list):           # If country list
        # Filter countries that exist in data
        eligible_countries = [c for c in countries if c in runner.all_countries]
        if len(eligible_countries) != len(countries):
            missing = [c for c in countries if c not in runner.all_countries]
            print(f"⚠️ Warning: Following countries not in data: {missing}")
        print(f"🎯 Will run experiments on {len(eligible_countries)} specified countries")
        
    else:                                       # Parameter type error
        raise ValueError("countries parameter must be string, list, or None")
    
    # Check if there are eligible countries
    if not eligible_countries:
        print("❌ No eligible countries to run experiments")
        return []
    
    # Further check sample size (for specified countries)
    if countries is not None:                   # If specified countries, need to check sample size
        valid_countries = []
        for country in eligible_countries:
            if runner.country_quality[country]['usable'] >= min_samples:
                valid_countries.append(country)
            else:
                print(f"⚠️ Skipping {country}: Insufficient samples ({runner.country_quality[country]['usable']} < {min_samples})")
        eligible_countries = valid_countries
    
    if not eligible_countries:
        print("❌ All specified countries have insufficient samples, cannot run experiments")
        return []
    
    # Display experiment plan
    print(f"📋 Experiment Plan:")
    print(f"   - Target countries: {eligible_countries}")
    print(f"   - Experiment types: {experiments}")
    print(f"   - Minimum samples: {min_samples}")
    
    # --- Execute experiment loop ---
    all_results = []                            # Store all experiment results
    
    for i, country in enumerate(eligible_countries, 1):
        print(f"\n[{i:2d}/{len(eligible_countries)}] Country: {country}")
        print("=" * 60)
        
        # Display data quality info for this country
        country_info = runner.country_quality[country]
        print(f"Country data: Total samples {country_info['total']:,}, Valid samples {country_info['usable']:,} ({country_info['usable_rate']:.1f}%)")
        
        country_results = []                    # Store all experiment results for this country
        
        # Run experiments based on experiments parameter
        if 'A' in experiments:                  # Experiment A: Self-Modeling
            result_A = runner.run_experiment_A(country, kwargs.get('n_trials_lgbm', 20))
            country_results.append(result_A)
        
        if 'B' in experiments:                  # Experiment B: Global Baseline
            result_B = runner.run_experiment_B(country, kwargs.get('n_trials_lgbm', 25))
            country_results.append(result_B)
        
        if 'B_dev' in experiments:              # Experiment B_dev: Developed Countries Baseline
            result_B_dev = runner.run_experiment_B_dev(country, kwargs.get('n_trials_lgbm', 25))
            country_results.append(result_B_dev)
        
        if 'C' in experiments:                  # Experiment C: Transfer Learning
            result_C = runner.run_experiment_C(
                country, 
                kwargs.get('n_finetune_trials', 25),
                kwargs.get('pretrain_epochs', 30),
                kwargs.get('finetune_epochs', 50)
            )
            country_results.append(result_C)
        
        if 'C_dev' in experiments:              # Experiment C_dev: Developed Countries Transfer Learning
            result_C_dev = runner.run_experiment_C_dev(
                country, 
                kwargs.get('n_finetune_trials', 25),
                kwargs.get('pretrain_epochs', 30),
                kwargs.get('finetune_epochs', 50)
            )
            country_results.append(result_C_dev)
        
        if 'D_lora' in experiments:             # Experiment D_lora: FT-Transformer LoRA
            result_D_lora = runner.run_experiment_D_lora(
                country, 
                source_mode='all',
                pretrain_epochs=kwargs.get('pretrain_epochs_ftt', 60),
                finetune_epochs=kwargs.get('finetune_epochs_ftt', 40),
                lora_rank=kwargs.get('lora_rank', 8),
                lora_alpha=kwargs.get('lora_alpha', 16)
            )
            country_results.append(result_D_lora)
        
        if 'D_lora_dev' in experiments:         # Experiment D_lora_dev: FT-Transformer LoRA Developed Countries
            result_D_lora_dev = runner.run_experiment_D_lora(
                country, 
                source_mode='dev',
                pretrain_epochs=kwargs.get('pretrain_epochs_ftt', 60),
                finetune_epochs=kwargs.get('finetune_epochs_ftt', 40),
                lora_rank=kwargs.get('lora_rank', 8),
                lora_alpha=kwargs.get('lora_alpha', 16)
            )
            country_results.append(result_D_lora_dev)
        
        if 'D_full' in experiments:             # Experiment D_full: FT-Transformer Full Fine-tuning
            result_D_full = runner.run_experiment_D_full(
                country, 
                source_mode='all',
                pretrain_epochs=kwargs.get('pretrain_epochs_ftt', 60),
                finetune_epochs=kwargs.get('finetune_epochs_ftt', 40)
            )
            country_results.append(result_D_full)
        
        if 'D_full_dev' in experiments:         # Experiment D_full_dev: FT-Transformer Full Fine-tuning Developed Countries
            result_D_full_dev = runner.run_experiment_D_full(
                country, 
                source_mode='dev',
                pretrain_epochs=kwargs.get('pretrain_epochs_ftt', 60),
                finetune_epochs=kwargs.get('finetune_epochs_ftt', 40)
            )
            country_results.append(result_D_full_dev)
        
        # Add this country's results to total results
        all_results.extend(country_results)
        
        # --- Display country experiment summary ---
        successful = [r for r in country_results if 'status' not in r and not np.isnan(r.get('R2_Average', np.nan))]
        failed = [r for r in country_results if 'status' in r or np.isnan(r.get('R2_Average', np.nan))]
        
        if successful:                          # If there are successful experiments
            avg_r2 = np.mean([r['R2_Average'] for r in successful])
            print(f"  📊 {country} Complete: Average R² = {avg_r2:.4f} ({len(successful)}/{len(country_results)} experiments successful)")
            
            # Display detailed results for each experiment
            for result in successful:
                exp_type = result['experiment_type']
                model_name = result['model']
                r2_score = result['R2_Average']
                print(f"     - {model_name}: R² = {r2_score:.4f}")
        else:                                   # If no successful experiments
            print(f"  ❌ {country} No successful experiments")
        
        # Display failed experiments (if any)
        if failed:
            print(f"  ⚠️ Failed experiments:")
            for result in failed:
                model_name = result.get('model', 'Unknown')
                error_msg = result.get('status', result.get('error_message', 'Unknown error'))
                print(f"     - {model_name}: {error_msg}")
    
    # --- Display global experiment summary ---
    print(f"\n{'='*80}")
    print(f"🎉 Global Experiments Complete!")
    
    # Count successful and failed experiments
    total_experiments = len(all_results)
    successful_experiments = len([r for r in all_results if 'status' not in r and not np.isnan(r.get('R2_Average', np.nan))])
    failed_experiments = total_experiments - successful_experiments
    
    print(f"📊 Experiment Statistics:")
    print(f"   - Total experiments: {total_experiments}")
    print(f"   - Successful experiments: {successful_experiments}")
    print(f"   - Failed experiments: {failed_experiments}")
    print(f"   - Success rate: {(successful_experiments/total_experiments)*100:.1f}%" if total_experiments > 0 else "   - Success rate: 0%")
    
    # Display results summary by experiment type
    if successful_experiments > 0:
        print(f"\n📈 Successful Experiment Results Summary:")
        exp_names = {
            'A': 'Self-Modeling', 
            'B': 'Global Baseline', 
            'B_dev': 'Developed Countries Baseline',
            'C': 'Transfer Learning',
            'C_dev': 'Developed Countries Transfer Learning',
            'D_lora': 'FTT-LoRA',
            'D_lora_dev': 'FTT-LoRA Developed Countries',
            'D_full': 'FTT Full Fine-tuning',
            'D_full_dev': 'FTT Full Fine-tuning Developed Countries'
        }
        for exp_type in experiments:
            exp_name = exp_names.get(exp_type, exp_type)
            # Match results containing experiment type in model name
            exp_results = [r for r in all_results 
                          if exp_type in r.get('model', '') 
                          and not np.isnan(r.get('R2_Average', np.nan))]
            if exp_results:
                avg_r2 = np.mean([r['R2_Average'] for r in exp_results])
                print(f"   - Experiment {exp_type} ({exp_name}): {len(exp_results)} successful, Average R² = {avg_r2:.4f}")
    
    return all_results                          # Return all experiment results