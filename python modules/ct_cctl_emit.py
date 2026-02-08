# ==============================================================================
# CT-CCTL-EMIT: Cross-Time Contrastive Learning for Emission Prediction
# Cross-time contrastive learning model for emission prediction
# ==============================================================================
"""
Core idea:
1. Pretraining: cross-time contrastive learning to capture time-invariant firm
   representations.
   - Positive pairs: same firm (gvkey) across different years.
   - Negative pairs: different firms.
2. Fine-tuning: freeze the encoder and train only the regression head.

Motivation:
- CCTL (KDD): cross-domain collaborative contrastive learning.
- CCDR (KDD): contrastive learning for domain-invariant representations.

Our contributions:
- Recast "cross-domain" contrastive learning as "cross-time" contrastive learning.
- Design positive/negative sampling for longitudinal panel data.
- Apply transfer learning to corporate emission prediction for the first time.

Enhancements:
- Checkpointing for resume/recovery.
- AutoML via Optuna-based hyperparameter search.
"""

import math
import random
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm

# Try to import scikit-learn. If unavailable, fall back to lightweight NumPy implementations.
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    def train_test_split(indices, test_size=0.25, random_state=42):
        """Minimal replacement for sklearn.model_selection.train_test_split (indices only)."""
        rng = np.random.RandomState(random_state)
        indices = np.asarray(list(indices))
        rng.shuffle(indices)
        n = len(indices)
        n_test = int(round(n * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return train_idx, test_idx

    class StandardScaler:
        """Minimal replacement for sklearn.preprocessing.StandardScaler (NumPy only)."""
        def __init__(self, with_mean=True, with_std=True, eps=1e-12):
            self.with_mean = with_mean
            self.with_std = with_std
            self.eps = eps
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1], dtype=np.float32)
            if self.with_std:
                std = X.std(axis=0)
                self.scale_ = np.where(std < self.eps, 1.0, std).astype(np.float32)
            else:
                self.scale_ = np.ones(X.shape[1], dtype=np.float32)
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean(np.abs(y_true - y_pred)))

    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean((y_true - y_pred) ** 2))

    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.size < 2:
            return float('nan')
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        if ss_tot == 0:
            return float('nan')
        return float(1.0 - ss_res / ss_tot)

# Try to import Optuna (for AutoML hyperparameter tuning).
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna is not installed; AutoML is unavailable. Run: pip install optuna.")


def _compute_nan_aware_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute per-target R²/MAE with NaN-aware masking and an overall average.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    results: Dict[str, float] = {}
    r2_scores: List[float] = []
    mae_scores: List[float] = []

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    for i in range(y_true.shape[1]):
        valid_mask = ~np.isnan(y_true[:, i])
        if valid_mask.sum() > 0:
            r2 = r2_score(y_true[valid_mask, i], y_pred[valid_mask, i])
            mae = mean_absolute_error(y_true[valid_mask, i], y_pred[valid_mask, i])
            results[f'R2_target_{i}'] = r2
            results[f'MAE_target_{i}'] = mae
            r2_scores.append(r2)
            mae_scores.append(mae)

    results['R2_Average'] = float(np.mean(r2_scores)) if r2_scores else float('nan')
    results['MAE_Average'] = float(np.mean(mae_scores)) if mae_scores else float('nan')
    return results


def _ridge_calibrate_predictions(
    preds_train: np.ndarray,
    y_train: np.ndarray,
    preds_val: np.ndarray,
    y_val: np.ndarray,
    preds_test: np.ndarray,
    lambdas: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Post-hoc ridge calibration / stacking on top of model predictions.

    We fit a ridge regressor (per target) using the prediction vector as features:
        y_j ≈ [pred_1, ..., pred_T, 1] · w_j
    Hyperparameter λ is selected on the validation set (no test leakage), then we refit
    on train+val and apply to test.
    """
    preds_train = np.asarray(preds_train, dtype=np.float64)
    preds_val = np.asarray(preds_val, dtype=np.float64)
    preds_test = np.asarray(preds_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)

    if lambdas is None:
        lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    n_targets = y_train.shape[1]
    n_features = preds_train.shape[1]

    def _fit_one(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
        # Add intercept term; do not penalize intercept.
        A1 = np.concatenate([A, np.ones((A.shape[0], 1), dtype=np.float64)], axis=1)
        reg = np.eye(A1.shape[1], dtype=np.float64) * lam
        reg[-1, -1] = 0.0
        w = np.linalg.solve(A1.T @ A1 + reg, A1.T @ b)
        return w

    def _predict(A: np.ndarray, w: np.ndarray) -> np.ndarray:
        A1 = np.concatenate([A, np.ones((A.shape[0], 1), dtype=np.float64)], axis=1)
        return A1 @ w

    # Select lambda by mean validation R² across targets (NaN-aware).
    best_lam = lambdas[0]
    best_score = -float('inf')

    for lam in lambdas:
        per_target_r2 = []
        for j in range(n_targets):
            mask_tr = ~np.isnan(y_train[:, j])
            mask_va = ~np.isnan(y_val[:, j])
            if mask_tr.sum() < 5 or mask_va.sum() < 5:
                continue
            w = _fit_one(preds_train[mask_tr], y_train[mask_tr, j], lam)
            pred_val_j = _predict(preds_val[mask_va], w)
            per_target_r2.append(r2_score(y_val[mask_va, j], pred_val_j))
        if per_target_r2:
            score = float(np.mean(per_target_r2))
            if score > best_score:
                best_score = score
                best_lam = lam

    # Refit on train+val with the chosen lambda.
    preds_trva = np.concatenate([preds_train, preds_val], axis=0)
    y_trva = np.concatenate([y_train, y_val], axis=0)

    W = np.zeros((n_features + 1, n_targets), dtype=np.float64)
    for j in range(n_targets):
        mask = ~np.isnan(y_trva[:, j])
        if mask.sum() < 5:
            # fallback: identity mapping for this target
            w = np.zeros((n_features + 1,), dtype=np.float64)
            if j < n_features:
                w[j] = 1.0
            W[:, j] = w
            continue
        w = _fit_one(preds_trva[mask], y_trva[mask, j], best_lam)
        W[:, j] = w

    # Apply to test
    preds_test_cal = np.zeros((preds_test.shape[0], n_targets), dtype=np.float64)
    for j in range(n_targets):
        preds_test_cal[:, j] = _predict(preds_test, W[:, j])

    info = {
        'ensemble_strategy': 'ridge_calibration',
        'selected_lambda': best_lam,
        'val_mean_r2_for_lambda_selection': best_score,
    }
    return preds_test_cal, info


# ==============================================================================
# 1. FT-Transformer encoder (base encoder).
# ==============================================================================

class NumericalEmbedding(nn.Module):
    """Numerical feature embedding: project each feature to d_model."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_features, d_model))
        self.biases = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in = self.weights.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        # output: (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weights + self.biases


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class LoRAAdapter(nn.Module):
    """
    Lightweight LoRA adapter: low-rank update (x + scale * xAB).

    This applies a LoRA-style increment to the activation/residual branch for
    parameter-efficient fine-tuning.
    """

    def __init__(self, d_model: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        rank = max(1, int(rank))
        alpha = int(alpha)
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(d_model, rank))
        self.B = nn.Parameter(torch.empty(rank, d_model))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        update = x @ self.A @ self.B
        return x + update * self.scale


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_lora = use_lora
        self.lora_attn = LoRAAdapter(d_model, lora_rank, lora_alpha) if use_lora else None
        self.lora_ffn = LoRAAdapter(d_model, lora_rank, lora_alpha) if use_lora else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.dropout(self.attn(self.norm1(x)))
        if self.use_lora and self.lora_attn is not None:
            attn_out = self.lora_attn(attn_out)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        if self.use_lora and self.lora_ffn is not None:
            ffn_out = self.lora_ffn(ffn_out)
        x = x + ffn_out
        return x


class FTTransformerEncoder(nn.Module):
    """
    FT-Transformer encoder.
    Encode tabular data into a fixed-dimensional representation.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ffn: int = 256,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Feature embedding
        self.feature_embedding = NumericalEmbedding(n_features, d_model)

        # [CLS] token for global pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features) input features
        Returns:
            (batch, d_model) CLS representation
        """
        batch_size = x.size(0)

        # Feature embedding: (batch, n_features, d_model)
        x = self.feature_embedding(x)

        # Add CLS token: (batch, 1 + n_features, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        for block in self.transformer_blocks:
            x = block(x)

        # Use CLS token output as the global representation
        x = self.norm(x[:, 0])  # (batch, d_model)
        return x


# ==============================================================================
# 2. Projection head (for contrastive learning)
# ==============================================================================

class ProjectionHead(nn.Module):
    """
    Projection head: map encoder output to the contrastive space.
    SimCLR-style: two-layer MLP + normalization.
    """
    def __init__(self, d_input: int, d_hidden: int = 128, d_output: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # L2 normalization for cosine similarity
        return F.normalize(x, dim=-1)


# ==============================================================================
# 3. Regression head
# ==============================================================================

class RegressionHead(nn.Module):
    """Multi-target regression head."""
    def __init__(self, d_input: int, n_targets: int, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# 4. Full CT-CCTL-EMIT model
# ==============================================================================

class CT_CCTL_EMIT(nn.Module):
    """
    Cross-Time Contrastive Learning for Emission Prediction
    Cross-time contrastive learning model for emission prediction.

    Two-stage training:
    1. Pretraining: contrastive learning (encoder + projection head).
    2. Fine-tuning: regression (freeze encoder, use regression head).
    """
    def __init__(
        self,
        n_features: int,
        n_targets: int = 6,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ffn: int = 256,
        dropout: float = 0.1,
        d_proj_hidden: int = 128,
        d_proj_output: int = 64,
        d_reg_hidden: int = 128,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()

        # Encoder
        self.encoder = FTTransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Contrastive projection head
        self.projection_head = ProjectionHead(
            d_input=d_model,
            d_hidden=d_proj_hidden,
            d_output=d_proj_output
        )

        # Regression head
        self.regression_head = RegressionHead(
            d_input=d_model,
            n_targets=n_targets,
            d_hidden=d_reg_hidden,
            dropout=dropout
        )

        self.d_model = d_model
        self.n_features = n_features
        self.n_targets = n_targets
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder representations."""
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return contrastive projections."""
        h = self.encode(x)
        return self.projection_head(h)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression predictions."""
        h = self.encode(x)
        return self.regression_head(h)

    def forward(self, x: torch.Tensor, mode: str = 'predict') -> torch.Tensor:
        """
        Args:
            x: (batch, n_features)
            mode: 'encode' | 'project' | 'predict'
        """
        if mode == 'encode':
            return self.encode(x)
        elif mode == 'project':
            return self.project(x)
        elif mode == 'predict':
            return self.predict(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def freeze_encoder(self):
        """Freeze encoder parameters (for fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.projection_head.parameters():
            param.requires_grad = False
        print("✅ Encoder and projection head are frozen.")

    def freeze_base_train_lora(self):
        """
        Freeze the encoder backbone and train only LoRA parameters (and the
        regression head).

        Used to construct the parameter-efficient transfer variant D_lora + CT:
        - Pretraining: contrastive learning (encoder + projection)
        - Fine-tuning: freeze encoder base; train LoRA in encoder + regression head
        """
        for name, param in self.encoder.named_parameters():
            # Keep only LoRA parameters trainable
            param.requires_grad = ('lora_' in name)
        for param in self.projection_head.parameters():
            param.requires_grad = False
        print("✅ Encoder base frozen; LoRA trainable; projection head frozen.")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection_head.parameters():
            param.requires_grad = True
        print("✅ Encoder and projection head are unfrozen.")


# ==============================================================================
# 5. Cross-time contrastive loss
# ==============================================================================

class CrossTimeContrastiveLoss(nn.Module):
    """
    Cross-time contrastive loss (InfoNCE variant).

    Core idea:
    - Same firm (gvkey) across years should have similar representations -> positive pairs
    - Different firms should be dissimilar -> negative pairs

    Inspired by CCDR contrastive loss and SimCLR InfoNCE.
    We adapt cross-domain contrast to cross-time for longitudinal panel data.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z_anchor: (batch, d) anchor projections
            z_positive: (batch, d) positive projections (same firm across years)
            z_negatives: (batch, n_neg, d) negative projections (batch negatives if None)

        Returns:
            contrastive loss
        """
        batch_size = z_anchor.size(0)

        # Positive similarity
        pos_sim = F.cosine_similarity(z_anchor, z_positive, dim=-1)  # (batch,)
        pos_sim = pos_sim / self.temperature

        if z_negatives is not None:
            # Explicit negatives
            # z_anchor: (batch, d) -> (batch, 1, d)
            # z_negatives: (batch, n_neg, d)
            neg_sim = F.cosine_similarity(
                z_anchor.unsqueeze(1),
                z_negatives,
                dim=-1
            )  # (batch, n_neg)
            neg_sim = neg_sim / self.temperature

            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1+n_neg)
            labels = torch.zeros(batch_size, dtype=torch.long, device=z_anchor.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # Batch negatives (SimCLR-style)
            # Similarity matrix for all pairs
            sim_matrix = F.cosine_similarity(
                z_anchor.unsqueeze(1),
                z_positive.unsqueeze(0),
                dim=-1
            ) / self.temperature  # (batch, batch)

            # Diagonal entries are positives
            labels = torch.arange(batch_size, device=z_anchor.device)
            loss = F.cross_entropy(sim_matrix, labels)

        return loss


# ==============================================================================
# 6. Datasets and samplers
# ==============================================================================

class CrossTimeContrastiveDataset(Dataset):
    """
    Cross-time contrastive learning dataset.

    Builds positive pairs for each sample (same firm across years).
    """
    def __init__(
        self,
        X: np.ndarray,
        gvkeys: np.ndarray,
        years: np.ndarray,
        y: Optional[np.ndarray] = None
    ):
        """
        Args:
            X: (N, F) feature matrix
            gvkeys: (N,) firm ID
            years: (N,) years
            y: (N, T) targets (optional, for fine-tuning)
        """
        self.X = torch.FloatTensor(X)
        self.gvkeys = gvkeys
        self.years = years
        self.y = torch.FloatTensor(y) if y is not None else None

        # Map firm IDs to sample indices
        self.gvkey_to_indices: Dict[Any, List[int]] = {}
        for idx, gvkey in enumerate(gvkeys):
            if gvkey not in self.gvkey_to_indices:
                self.gvkey_to_indices[gvkey] = []
            self.gvkey_to_indices[gvkey].append(idx)

        # Keep firms with multiple years (eligible for positive pairs)
        self.valid_indices = []
        for gvkey, indices in self.gvkey_to_indices.items():
            if len(indices) >= 2:  # at least two years
                self.valid_indices.extend(indices)

        print("📊 Dataset summary:")
        print(f"   - Total samples: {len(X)}")
        print(f"   - Eligible for contrastive learning: {len(self.valid_indices)}")
        print(
            f"   - Firms with multi-year data: "
            f"{len([k for k, v in self.gvkey_to_indices.items() if len(v) >= 2])}"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Anchor sample
        anchor_idx = self.valid_indices[idx]
        anchor_x = self.X[anchor_idx]
        anchor_gvkey = self.gvkeys[anchor_idx]
        anchor_year = self.years[anchor_idx]

        # Randomly select a positive from the same firm (other year)
        same_company_indices = self.gvkey_to_indices[anchor_gvkey]
        positive_candidates = [i for i in same_company_indices if i != anchor_idx]

        if positive_candidates:
            positive_idx = random.choice(positive_candidates)
        else:
            # Fallback to self if only one sample exists (not ideal)
            positive_idx = anchor_idx

        positive_x = self.X[positive_idx]

        result = {
            'anchor_x': anchor_x,
            'positive_x': positive_x,
            'anchor_idx': anchor_idx,
            'positive_idx': positive_idx,
        }

        if self.y is not None:
            result['anchor_y'] = self.y[anchor_idx]

        return result


class SimpleRegressionDataset(Dataset):
    """Simple regression dataset (for fine-tuning)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ==============================================================================
# 7. Trainer
# ==============================================================================

class CT_CCTL_EMIT_Trainer:
    """
    CT-CCTL-EMIT model trainer.

    Two-stage training:
    1. Pretraining (contrastive): learn time-invariant firm representations on source data.
    2. Fine-tuning (regression): train regression head on target data.

    Enhancements:
    - Checkpointing for resume/recovery.
    - Detailed loss tracking for train/val.
    """
    def __init__(
        self,
        model: CT_CCTL_EMIT,
        device: str = 'cuda',
        temperature: float = 0.07,
        checkpoint_dir: Optional[str] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.contrastive_loss = CrossTimeContrastiveLoss(temperature=temperature)
        
        # Checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        self.training_state = {
            'stage': None,  # 'pretrain' or 'finetune'
            'epoch': 0,
            'best_metric': None
        }

    def _save_checkpoint(
        self,
        stage: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        history: Dict,
        best_state: Optional[Dict] = None,
        extra_info: Optional[Dict] = None
    ):
        """Save a checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'stage': stage,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'best_state': best_state,
            'timestamp': datetime.now().isoformat(),
            'extra_info': extra_info or {}
        }
        
        ckpt_path = self.checkpoint_dir / f'{stage}_checkpoint.pt'
        torch.save(checkpoint, ckpt_path)
        
        # Also save an epoch-tagged checkpoint for rollback
        if epoch % 10 == 0:  # every 10 epochs
            ckpt_path_epoch = self.checkpoint_dir / f'{stage}_epoch_{epoch}.pt'
            torch.save(checkpoint, ckpt_path_epoch)

    def _load_checkpoint(self, stage: str) -> Optional[Dict]:
        """Load a checkpoint."""
        if self.checkpoint_dir is None:
            return None
        
        ckpt_path = self.checkpoint_dir / f'{stage}_checkpoint.pt'
        if not ckpt_path.exists():
            return None
        
        print(f"📂 Checkpoint found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        return checkpoint

    def _build_loader_kwargs(
        self,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = 2,
    ) -> Dict[str, Any]:
        num_workers = int(num_workers or 0)
        if pin_memory is None:
            pin_memory = str(self.device).startswith('cuda')

        kwargs: Dict[str, Any] = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            kwargs['persistent_workers'] = bool(persistent_workers)
            if prefetch_factor is not None:
                kwargs['prefetch_factor'] = int(prefetch_factor)
        return kwargs

    def pretrain(
        self,
        X_source: np.ndarray,
        gvkeys_source: np.ndarray,
        years_source: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        gvkeys_val: Optional[np.ndarray] = None,
        years_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        resume: bool = True,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: Optional[bool] = None,
        dataloader_persistent_workers: bool = False,
        dataloader_prefetch_factor: Optional[int] = 2,
    ) -> Dict[str, List[float]]:
        """
        Pretraining stage: cross-time contrastive learning.

        Args:
            X_source: source features
            gvkeys_source: source firm IDs
            years_source: source years
            X_val, gvkeys_val, years_val: validation data (optional)
            epochs: number of epochs
            batch_size: batch size
            lr: learning rate
            weight_decay: weight decay
            verbose: whether to log progress
            resume: resume from checkpoint if available

        Returns:
            training history (train_loss, val_loss)
        """
        print("="*60)
        print("🚀 Starting pretraining: cross-time contrastive learning")
        print("="*60)

        # Prepare training data
        train_dataset = CrossTimeContrastiveDataset(X_source, gvkeys_source, years_source)
        loader_kwargs = self._build_loader_kwargs(
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
            persistent_workers=dataloader_persistent_workers,
            prefetch_factor=dataloader_prefetch_factor,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
            drop_last=True
        )
        
        # Prepare validation data (if provided)
        val_loader = None
        if X_val is not None and gvkeys_val is not None and years_val is not None:
            val_dataset = CrossTimeContrastiveDataset(X_val, gvkeys_val, years_val)
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    **loader_kwargs,
                    drop_last=False
                )
                print(f"📊 Validation samples: {len(val_dataset)}")

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )

        history = {'train_loss': [], 'val_loss': []}
        start_epoch = 0
        
        # Try to resume from checkpoint
        if resume:
            checkpoint = self._load_checkpoint('pretrain')
            if checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                history = checkpoint['history']
                start_epoch = checkpoint['epoch'] + 1
                print(f"✅ Resuming from epoch {start_epoch}")
        
        self.model.train()

        for epoch in range(start_epoch, epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", disable=not verbose)
            for batch in pbar:
                anchor_x = batch['anchor_x'].to(self.device)
                positive_x = batch['positive_x'].to(self.device)

                # Projection representations
                z_anchor = self.model.project(anchor_x)
                z_positive = self.model.project(positive_x)

                # Contrastive loss
                loss = self.contrastive_loss(z_anchor, z_positive)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})

            scheduler.step()
            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase (if available)
            avg_val_loss = np.nan
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        anchor_x = batch['anchor_x'].to(self.device)
                        positive_x = batch['positive_x'].to(self.device)
                        
                        z_anchor = self.model.project(anchor_x)
                        z_positive = self.model.project(positive_x)
                        loss = self.contrastive_loss(z_anchor, z_positive)
                        
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / max(n_val_batches, 1)
                history['val_loss'].append(avg_val_loss)
            
            # Save checkpoint
            self._save_checkpoint(
                stage='pretrain',
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history
            )

            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss = {avg_val_loss:.4f}" if not np.isnan(avg_val_loss) else ""
                print(f"  Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}{val_str}, LR = {scheduler.get_last_lr()[0]:.6f}")

        final_train_loss = history['train_loss'][-1] if history['train_loss'] else np.nan
        print(f"✅ Pretraining completed. Final train loss: {final_train_loss:.4f}")
        return history

    def finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 20,
        freeze_encoder: bool = True,
        finetune_strategy: Optional[str] = None,  # 'head_only' | 'lora' | 'full'
        weight_decay: float = 1e-4,
        verbose: bool = True,
        resume: bool = True,
        checkpoint_stage: Optional[str] = None,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: Optional[bool] = None,
        dataloader_persistent_workers: bool = False,
        dataloader_prefetch_factor: Optional[int] = 2,
    ) -> Dict[str, List[float]]:
        """
        Fine-tuning stage: regression prediction.

        Args:
            X_train, y_train: target-domain training data
            X_val, y_val: validation data (optional, for early stop/model selection)
            dataloader_num_workers: number of DataLoader workers
            dataloader_pin_memory: whether to pin memory (None=auto)
            dataloader_persistent_workers: keep workers alive between epochs
            dataloader_prefetch_factor: DataLoader prefetch factor
            freeze_encoder: whether to freeze encoder
            weight_decay: weight decay
            verbose: whether to log progress
            resume: resume from checkpoint if available

        Returns:
            training history (train_loss, val_loss, val_r2)

        Note:
            - train_loss: train MSE
            - val_loss: validation MSE (for early stopping)
            - val_r2: validation R^2 (for monitoring)
        """
        print("="*60)
        print("🎯 Starting fine-tuning: regression prediction")
        print("   📝 Loss note: train_loss=train, val_loss/val_r2=validation")
        print("="*60)

        # Freeze encoder
        if finetune_strategy is None:
            finetune_strategy = 'head_only' if freeze_encoder else 'full'

        if finetune_strategy not in ('head_only', 'lora', 'full'):
            raise ValueError(f"Unknown finetune_strategy: {finetune_strategy}")

        if finetune_strategy == 'head_only':
            self.model.freeze_encoder()
        elif finetune_strategy == 'lora':
            self.model.freeze_base_train_lora()
        else:
            # full
            self.model.unfreeze_encoder()

        # Prepare data
        train_dataset = SimpleRegressionDataset(X_train, y_train)
        loader_kwargs = self._build_loader_kwargs(
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
            persistent_workers=dataloader_persistent_workers,
            prefetch_factor=dataloader_prefetch_factor,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = SimpleRegressionDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )

        # Select trainable parameters
        if finetune_strategy == 'head_only':
            params = self.model.regression_head.parameters()
        elif finetune_strategy == 'lora':
            # Regression head + LoRA in encoder
            lora_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            params = list(self.model.regression_head.parameters()) + lora_params
        else:
            params = self.model.parameters()

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 2
        )

        history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        start_epoch = 0
        
        # Checkpoint stage (avoid overwriting across head_only/lora/full)
        if checkpoint_stage is None:
            checkpoint_stage = f'finetune_{finetune_strategy}'

        # Try to resume from checkpoint (only if stage matches)
        if resume:
            checkpoint = self._load_checkpoint(checkpoint_stage)
            if checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # Optimizer/scheduler groups may be incompatible across strategies
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if checkpoint['scheduler_state_dict']:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"⚠️ Incompatible optimizer/scheduler state in checkpoint; skipped: {e}")
                history = checkpoint['history']
                start_epoch = checkpoint['epoch'] + 1
                best_state = checkpoint.get('best_state')
                best_val_loss = checkpoint.get('extra_info', {}).get('best_val_loss', float('inf'))
                patience_counter = checkpoint.get('extra_info', {}).get('patience_counter', 0)
                print(f"✅ Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Handle NaN targets
                valid_mask = ~torch.isnan(y_batch)
                if valid_mask.sum() == 0:
                    continue

                pred = self.model.predict(X_batch)
                loss = F.mse_loss(pred[valid_mask], y_batch[valid_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)
            history['train_loss'].append(avg_train_loss)

            # Validate
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        pred = self.model.predict(X_batch)
                        valid_mask = ~torch.isnan(y_batch)
                        if valid_mask.sum() > 0:
                            val_loss += F.mse_loss(pred[valid_mask], y_batch[valid_mask]).item()
                            all_preds.append(pred.cpu().numpy())
                            all_targets.append(y_batch.cpu().numpy())

                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)

                # Compute R^2
                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)
                valid_mask = ~np.isnan(all_targets)
                if valid_mask.sum() > 0:
                    val_r2 = r2_score(
                        all_targets[valid_mask],
                        all_preds[valid_mask]
                    )
                else:
                    val_r2 = np.nan
                history['val_r2'].append(val_r2)

                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save checkpoint
                self._save_checkpoint(
                    stage=checkpoint_stage,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history,
                    best_state=best_state,
                    extra_info={
                        'best_val_loss': best_val_loss,
                        'patience_counter': patience_counter
                    }
                )

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}, Val R² = {val_r2:.4f}")

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                # Save checkpoint even without validation
                self._save_checkpoint(
                    stage=checkpoint_stage,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history
                )
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        final_val_r2 = history['val_r2'][-1] if history['val_r2'] else np.nan
        print(f"✅ Fine-tuning complete. Best val R^2: {final_val_r2:.4f}")
        return history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Returns:
            dict of metrics
        """
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            preds = self.model.predict(X_test_t).cpu().numpy()

        return _compute_nan_aware_metrics(y_test, preds)


# ==============================================================================
# 8. Convenience experiment runner
# ==============================================================================

def run_ct_cctl_emit_experiment(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    target_country: str,
    df_pretrain: Optional[pd.DataFrame] = None,
    df_finetune: Optional[pd.DataFrame] = None,
    source_countries: Optional[List[str]] = None,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 100,
    pretrain_batch_size: int = 256,
    finetune_batch_size: int = 64,
    d_model: int = 192,
    n_heads: int = 8,
    n_layers: int = 3,
    temperature: float = 0.07,
    freeze_encoder: bool = True,
    finetune_strategy: Optional[str] = None,  # 'head_only' | 'lora' | 'full'
    lora_rank: int = 8,
    lora_alpha: int = 16,
    random_state: int = 42,
    device: str = 'cuda',
    verbose: bool = True,
    checkpoint_dir: Optional[str] = None,
    ensemble_strategy: Optional[str] = "ridge_calibration",
    dataloader_num_workers: int = 0,
    dataloader_pin_memory: Optional[bool] = None,
    dataloader_persistent_workers: bool = False,
    dataloader_prefetch_factor: Optional[int] = 2,
) -> Dict[str, Any]:
    """
    Run a CT-CCTL-EMIT experiment.

    Args:
        df: full dataset
        features: feature column names
        targets: target column names
        target_country: target country
        source_countries: source countries (None = all except target)
        pretrain_epochs: pretraining epochs
        finetune_epochs: fine-tuning epochs
        d_model: model dimension
        n_heads: number of attention heads
        n_layers: number of Transformer layers
        temperature: contrastive temperature
        freeze_encoder: whether to freeze encoder during fine-tuning
        random_state: random seed
        device: device
        verbose: verbose logging

    Returns:
        result dict
    """
    print(f"\n{'='*70}")
    print(f"🌍 CT-CCTL-EMIT experiment: target country = {target_country}")
    print(f"{'='*70}")

    # Set random seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    # Allow "self-supervised pretraining on full data" and "supervised fine-tuning
    # on labeled subset":
    # - df_pretrain: contrastive data (features + gvkey + fiscalyear + loc only)
    # - df_finetune: target-domain supervised fine-tuning (targets required)
    df_pretrain_used = df_pretrain if df_pretrain is not None else df
    df_finetune_used = df_finetune if df_finetune is not None else df

    # Prepare source data
    if source_countries is None:
        source_countries = [c for c in df_pretrain_used['loc'].unique() if c != target_country]

    df_source = df_pretrain_used[df_pretrain_used['loc'].isin(source_countries)].copy()
    df_target = df_finetune_used[df_finetune_used['loc'] == target_country].copy()

    print("📊 Data summary:")
    print(f"   - #source countries: {len(source_countries)}")
    print(f"   - #source samples: {len(df_source)}")
    print(f"   - #target samples: {len(df_target)}")

    # Feature standardization (fit on source)
    scaler = StandardScaler()
    X_source = scaler.fit_transform(df_source[features].values)
    X_target = scaler.transform(df_target[features].values)
    y_target = df_target[targets].values

    gvkeys_source = df_source['gvkey'].values
    years_source = df_source['fiscalyear'].values

    # Target split (time-based)
    years_target = df_target['fiscalyear'].values
    unique_years = sorted(df_target['fiscalyear'].unique())

    if len(unique_years) >= 3:
        train_years = unique_years[:-2]
        val_years = [unique_years[-2]]
        test_years = [unique_years[-1]]
    else:
        # Too few samples; fallback to random split
        train_idx, test_idx = train_test_split(
            range(len(X_target)), test_size=0.3, random_state=random_state
        )
        val_idx = test_idx[:len(test_idx)//2]
        test_idx = test_idx[len(test_idx)//2:]

        X_train = X_target[train_idx]
        X_val = X_target[val_idx]
        X_test = X_target[test_idx]
        y_train = y_target[train_idx]
        y_val = y_target[val_idx]
        y_test = y_target[test_idx]

        print("   - Random split (insufficient years)")
        print(f"   - Train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

        train_years = val_years = test_years = None

    if train_years is not None:
        train_mask = np.isin(years_target, train_years)
        val_mask = np.isin(years_target, val_years)
        test_mask = np.isin(years_target, test_years)

        X_train, y_train = X_target[train_mask], y_target[train_mask]
        X_val, y_val = X_target[val_mask], y_target[val_mask]
        X_test, y_test = X_target[test_mask], y_target[test_mask]

        print("   - Time-based split:")
        print(f"     Train years: {train_years}, samples: {len(X_train)}")
        print(f"     Val years: {val_years}, samples: {len(X_val)}")
        print(f"     Test years: {test_years}, samples: {len(X_test)}")

    # Build model
    model = CT_CCTL_EMIT(
        n_features=len(features),
        n_targets=len(targets),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        use_lora=(finetune_strategy == 'lora'),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Set checkpoint directory
    ckpt_dir = None
    if checkpoint_dir:
        ckpt_dir = os.path.join(checkpoint_dir, f'cctl_{target_country}')
    
    trainer = CT_CCTL_EMIT_Trainer(
        model, 
        device=device, 
        temperature=temperature,
        checkpoint_dir=ckpt_dir
    )

    # Stage 1: pretraining
    pretrain_history = trainer.pretrain(
        X_source=X_source,
        gvkeys_source=gvkeys_source,
        years_source=years_source,
        epochs=pretrain_epochs,
        batch_size=pretrain_batch_size,
        verbose=verbose,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
    )

    # Stage 2: fine-tuning
    finetune_history = trainer.finetune(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=finetune_epochs,
        batch_size=finetune_batch_size,
        freeze_encoder=freeze_encoder,
        finetune_strategy=finetune_strategy,
        verbose=verbose,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
    )

    # Evaluate (raw)
    test_results_raw = trainer.evaluate(X_test, y_test)

    # Optional: post-hoc ensemble / calibration (NO test leakage: hyperparameters selected on val only)
    ensemble_info = None
    test_results = dict(test_results_raw)
    if ensemble_strategy in ("ridge_calibration", "ridge"):
        try:
            # Predict on train/val/test
            trainer.model.eval()
            with torch.no_grad():
                preds_train = trainer.model.predict(torch.FloatTensor(X_train).to(trainer.device)).cpu().numpy()
                preds_val = trainer.model.predict(torch.FloatTensor(X_val).to(trainer.device)).cpu().numpy()
                preds_test = trainer.model.predict(torch.FloatTensor(X_test).to(trainer.device)).cpu().numpy()

            preds_test_cal, ensemble_info = _ridge_calibrate_predictions(
                preds_train=preds_train,
                y_train=y_train,
                preds_val=preds_val,
                y_val=y_val,
                preds_test=preds_test,
            )
            test_results_cal = _compute_nan_aware_metrics(y_test, preds_test_cal)
            # Use calibrated results as the primary reported metrics (still keep raw in output).
            test_results = test_results_cal
        except Exception as e:
            if verbose:
                print(f"⚠️ Ensemble calibration failed; using raw model outputs. Reason: {str(e)[:200]}")
            ensemble_info = {'ensemble_strategy': ensemble_strategy, 'status': 'failed', 'error': str(e)}

    # Assemble results
    result = {
        'model': 'CT-CCTL-EMIT',
        'country': target_country,
        'experiment_type': 'Cross_Time_Contrastive_Learning',
        'R2_Average': test_results.get('R2_Average', np.nan),
        'MAE_Average': test_results.get('MAE_Average', np.nan),
        'R2_Average_raw': test_results_raw.get('R2_Average', np.nan),
        'MAE_Average_raw': test_results_raw.get('MAE_Average', np.nan),
        'ensemble_strategy': ensemble_info.get('ensemble_strategy') if isinstance(ensemble_info, dict) else ensemble_strategy,
        'ensemble_selected_lambda': ensemble_info.get('selected_lambda') if isinstance(ensemble_info, dict) else None,
        'ensemble_val_mean_r2': ensemble_info.get('val_mean_r2_for_lambda_selection') if isinstance(ensemble_info, dict) else None,
        'pretrain_final_loss': pretrain_history['train_loss'][-1] if pretrain_history['train_loss'] else np.nan,
        'finetune_final_val_r2': finetune_history['val_r2'][-1] if finetune_history['val_r2'] else np.nan,
        'source_samples': len(X_source),
        'target_train_samples': len(X_train),
        'target_test_samples': len(X_test),
        'freeze_encoder': freeze_encoder,
        'd_model': d_model,
        'n_layers': n_layers,
        'temperature': temperature,
        # Add sample-size/year metadata (for analysis)
        'n_source': len(X_source),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_total': len(X_target),
        'years_min': int(years_target.min()) if len(years_target) > 0 else None,
        'years_max': int(years_target.max()) if len(years_target) > 0 else None,
        'n_years': len(np.unique(years_target)) if len(years_target) > 0 else 0,
    }

    # Add per-target R^2
    for k, v in test_results.items():
        if k.startswith('R2_target_'):
            result[k] = v
    for k, v in test_results_raw.items():
        if k.startswith('R2_target_'):
            result[f'{k}_raw'] = v

    print("\n📈 Final test results:")
    print(f"   - Mean R^2: {result['R2_Average']:.4f}")
    print(f"   - Mean MAE: {result['MAE_Average']:.4f}")
    if ensemble_info:
        print(f"   - Raw Mean R^2: {result['R2_Average_raw']:.4f}")

    return result


# ==============================================================================
# 9. AutoML hyperparameter optimization (Optuna)
# ==============================================================================

class CT_CCTL_EMIT_AutoML:
    """
    Optuna-based hyperparameter optimization.

    Objective: maximize validation R^2.

    Search space:
    - Architecture: d_model, n_heads, n_layers
    - Pretraining: pretrain_lr, temperature, pretrain_epochs
    - Fine-tuning: finetune_lr, finetune_batch_size, freeze_encoder
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        targets: List[str],
        target_country: str,
        source_countries: Optional[List[str]] = None,
        device: str = 'cuda',
        random_state: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,  # seconds
        study_name: Optional[str] = None,
        storage: Optional[str] = None,  # Optuna DB path
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Run: pip install optuna")
        
        self.df = df
        self.features = features
        self.targets = targets
        self.target_country = target_country
        self.source_countries = source_countries
        self.device = device
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f'cctl_emit_{target_country}'
        self.storage = storage
        
        # Preprocess data (avoid repeated work per trial)
        self._prepare_data()
    
    def _prepare_data(self):
        """Preprocess data."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        if self.source_countries is None:
            self.source_countries = [c for c in self.df['loc'].unique() if c != self.target_country]
        
        df_source = self.df[self.df['loc'].isin(self.source_countries)].copy()
        df_target = self.df[self.df['loc'] == self.target_country].copy()
        
        # Feature standardization
        scaler = StandardScaler()
        self.X_source = scaler.fit_transform(df_source[self.features].values)
        self.X_target = scaler.transform(df_target[self.features].values)
        self.scaler = scaler
        
        self.y_source = df_source[self.targets].values
        self.y_target = df_target[self.targets].values
        
        self.gvkeys_source = df_source['gvkey'].values
        self.years_source = df_source['fiscalyear'].values
        
        # Target time-based split
        years_target = df_target['fiscalyear'].values
        unique_years = sorted(df_target['fiscalyear'].unique())
        
        if len(unique_years) >= 3:
            train_years = unique_years[:-2]
            val_years = [unique_years[-2]]
            test_years = [unique_years[-1]]
            
            train_mask = np.isin(years_target, train_years)
            val_mask = np.isin(years_target, val_years)
            test_mask = np.isin(years_target, test_years)
            
            self.X_train, self.y_train = self.X_target[train_mask], self.y_target[train_mask]
            self.X_val, self.y_val = self.X_target[val_mask], self.y_target[val_mask]
            self.X_test, self.y_test = self.X_target[test_mask], self.y_target[test_mask]
        else:
            # Random split
            train_idx, test_idx = train_test_split(
                range(len(self.X_target)), test_size=0.3, random_state=self.random_state
            )
            val_idx = test_idx[:len(test_idx)//2]
            test_idx = test_idx[len(test_idx)//2:]
            
            self.X_train = self.X_target[train_idx]
            self.X_val = self.X_target[val_idx]
            self.X_test = self.X_target[test_idx]
            self.y_train = self.y_target[train_idx]
            self.y_val = self.y_target[val_idx]
            self.y_test = self.y_target[test_idx]
        
        print("📊 AutoML data prepared:")
        print(f"   - Source samples: {len(self.X_source)}")
        print(f"   - Target train/val/test: {len(self.X_train)}/{len(self.X_val)}/{len(self.X_test)}")
    
    def _objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        d_model = trial.suggest_categorical('d_model', [64, 128, 192, 256])
        n_heads = trial.suggest_categorical('n_heads', [4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 4)
        
        pretrain_lr = trial.suggest_float('pretrain_lr', 1e-4, 1e-2, log=True)
        pretrain_epochs = trial.suggest_int('pretrain_epochs', 20, 80, step=10)
        temperature = trial.suggest_float('temperature', 0.03, 0.2, log=True)
        
        finetune_lr = trial.suggest_float('finetune_lr', 1e-4, 1e-2, log=True)
        finetune_epochs = trial.suggest_int('finetune_epochs', 50, 150, step=25)
        finetune_batch_size = trial.suggest_categorical('finetune_batch_size', [32, 64, 128])
        freeze_encoder = trial.suggest_categorical('freeze_encoder', [True, False])
        
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
        
        try:
            # Build model
            model = CT_CCTL_EMIT(
                n_features=len(self.features),
                n_targets=len(self.targets),
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers
            )
            
            trainer = CT_CCTL_EMIT_Trainer(model, device=self.device, temperature=temperature)
            
            # Pretraining
            trainer.pretrain(
                X_source=self.X_source,
                gvkeys_source=self.gvkeys_source,
                years_source=self.years_source,
                epochs=pretrain_epochs,
                lr=pretrain_lr,
                verbose=False
            )
            
            # Fine-tuning
            finetune_history = trainer.finetune(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                epochs=finetune_epochs,
                batch_size=finetune_batch_size,
                lr=finetune_lr,
                freeze_encoder=freeze_encoder,
                verbose=False
            )
            
            # Use validation R^2 as objective
            val_r2 = finetune_history['val_r2'][-1] if finetune_history['val_r2'] else -1.0
            
            # Report intermediate result (for pruning)
            trial.report(val_r2, step=0)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return val_r2
            
        except Exception as e:
            print(f"⚠️ Trial {trial.number} failed: {e}")
            return -1.0
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        print("="*70)
        print("🔍 CT-CCTL-EMIT AutoML hyperparameter optimization")
        print(f"   Target country: {self.target_country}")
        print(f"   Total trials: {self.n_trials}")
        print(f"   Timeout: {self.timeout}s" if self.timeout else "   No timeout")
        print("="*70)
        
        # Create or load the study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Report best result
        print("\n" + "="*70)
        print("🏆 Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        print(f"\n   Best validation R^2: {study.best_value:.4f}")
        print("="*70)
        
        # Evaluate on the test set with best parameters
        best_params = study.best_params
        
        # Ensure d_model is divisible by n_heads
        d_model = best_params['d_model']
        n_heads = best_params['n_heads']
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
        
        model = CT_CCTL_EMIT(
            n_features=len(self.features),
            n_targets=len(self.targets),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=best_params['n_layers']
        )
        
        trainer = CT_CCTL_EMIT_Trainer(
            model, 
            device=self.device, 
            temperature=best_params['temperature']
        )
        
        # Retrain using best parameters
        trainer.pretrain(
            X_source=self.X_source,
            gvkeys_source=self.gvkeys_source,
            years_source=self.years_source,
            epochs=best_params['pretrain_epochs'],
            lr=best_params['pretrain_lr'],
            verbose=True
        )
        
        trainer.finetune(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=best_params['finetune_epochs'],
            batch_size=best_params['finetune_batch_size'],
            lr=best_params['finetune_lr'],
            freeze_encoder=best_params['freeze_encoder'],
            verbose=True
        )
        
        # Test-set evaluation
        test_results = trainer.evaluate(self.X_test, self.y_test)
        
        print("\n📈 Final test-set results:")
        print(f"   - Mean R^2: {test_results['R2_Average']:.4f}")
        print(f"   - Mean MAE: {test_results['MAE_Average']:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_val_r2': study.best_value,
            'test_results': test_results,
            'study': study
        }


def run_automl_experiment(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    target_country: str,
    source_countries: Optional[List[str]] = None,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    device: str = 'cuda',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Convenience wrapper for running AutoML (single-country).

    Args:
        df: full dataset
        features: feature column names
        targets: target column names
        target_country: target country
        source_countries: source countries
        n_trials: number of trials
        timeout: timeout (seconds)
        device: device
        random_state: random seed

    Returns:
        dict with best params and test results
    """
    automl = CT_CCTL_EMIT_AutoML(
        df=df,
        features=features,
        targets=targets,
        target_country=target_country,
        source_countries=source_countries,
        device=device,
        random_state=random_state,
        n_trials=n_trials,
        timeout=timeout
    )
    
    return automl.optimize()


# ==============================================================================
# 10. Multi-country joint AutoML search (Plan A)
# ==============================================================================

class CT_CCTL_EMIT_MultiCountryAutoML:
    """
    Joint hyperparameter optimization across multiple target countries.

    Plan A:
    - Objective: maximize the mean validation R^2 across all target countries.
    - Each trial runs on all (e.g., 55) countries and averages R^2 as the score.
    - The search yields best hyperparameters and per-country results.

    Advantages:
    - Finds hyperparameters that generalize across countries.
    - Produces experiment outputs as part of the search.
    - Improves global consistency of results.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        targets: List[str],
        target_countries: List[str],
        min_samples: int = 50,
        device: str = 'cuda',
        random_state: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        force_freeze_encoder: bool = False,
    ):
        """
        Args:
            df: full dataset
            features: feature column names
            targets: target column names
            target_countries: target country list (typically the 55 countries with >= 50 labeled samples)
            min_samples: minimum effective sample size (default: 50)
            device: device
            random_state: random seed
            n_trials: number of trials
            timeout: timeout (seconds)
            study_name: study name
            storage: Optuna DB path
            checkpoint_dir: checkpoint directory
            force_freeze_encoder: whether to force freezing the encoder (to mitigate catastrophic forgetting)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Run: pip install optuna")
        
        self.df = df
        self.features = features
        self.targets = targets
        self.device = device
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or 'cctl_emit_multi_country'
        self.storage = storage
        self.checkpoint_dir = checkpoint_dir
        self.min_samples = min_samples
        self.force_freeze_encoder = force_freeze_encoder
        
        # Filter eligible countries (samples >= min_samples)
        self.target_countries = self._filter_eligible_countries(target_countries)
        print(f"✅ Eligible countries: {len(self.target_countries)}")
        
        # Store per-country data
        self.country_data = {}
        
        # Store per-country results for the best trial
        self.best_trial_all_results = {}
        
        # Preprocess all countries
        self._prepare_all_data()
    
    def _filter_eligible_countries(self, countries: List[str]) -> List[str]:
        """Filter countries with effective sample size >= min_samples."""
        eligible = []
        for country in countries:
            df_country = self.df[self.df['loc'] == country]
            # Effective samples: rows where all targets are present
            valid_mask = df_country[self.targets].notna().all(axis=1)
            n_valid = valid_mask.sum()
            if n_valid >= self.min_samples:
                eligible.append(country)
        return eligible
    
    def _prepare_all_data(self):
        """Preprocess data for all countries."""
        print("\n📊 Preprocessing data for all countries...")
        
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Global scaler (fit on all data)
        self.global_scaler = StandardScaler()
        self.global_scaler.fit(self.df[self.features].values)
        
        for country in tqdm(self.target_countries, desc="Preparing data"):
            # Source = all countries except the current target
            source_countries = [c for c in self.df['loc'].unique() if c != country]
            
            df_source = self.df[self.df['loc'].isin(source_countries)].copy()
            df_target = self.df[self.df['loc'] == country].copy()
            
            # Standardize with the global scaler
            X_source = self.global_scaler.transform(df_source[self.features].values)
            X_target = self.global_scaler.transform(df_target[self.features].values)
            
            y_source = df_source[self.targets].values
            y_target = df_target[self.targets].values
            
            gvkeys_source = df_source['gvkey'].values
            years_source = df_source['fiscalyear'].values
            
            # Target time-based split
            years_target = df_target['fiscalyear'].values
            unique_years = sorted(df_target['fiscalyear'].unique())
            
            if len(unique_years) >= 3:
                train_years = unique_years[:-2]
                val_years = [unique_years[-2]]
                test_years = [unique_years[-1]]
                
                train_mask = np.isin(years_target, train_years)
                val_mask = np.isin(years_target, val_years)
                test_mask = np.isin(years_target, test_years)
                
                X_train, y_train = X_target[train_mask], y_target[train_mask]
                X_val, y_val = X_target[val_mask], y_target[val_mask]
                X_test, y_test = X_target[test_mask], y_target[test_mask]
            else:
                # Random split
                indices = list(range(len(X_target)))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.3, random_state=self.random_state
                )
                val_idx = test_idx[:len(test_idx)//2]
                test_idx = test_idx[len(test_idx)//2:]
                
                X_train = X_target[train_idx]
                X_val = X_target[val_idx]
                X_test = X_target[test_idx]
                y_train = y_target[train_idx]
                y_val = y_target[val_idx]
                y_test = y_target[test_idx]
            
            self.country_data[country] = {
                'X_source': X_source,
                'y_source': y_source,
                'gvkeys_source': gvkeys_source,
                'years_source': years_source,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
            }
        
        print(f"✅ Preprocessing complete: {len(self.country_data)} countries")
    
    def _train_single_country(
        self,
        country: str,
        params: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Train and evaluate on a single country."""
        data = self.country_data[country]
        
        # Ensure d_model is divisible by n_heads
        d_model = params['d_model']
        n_heads = params['n_heads']
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
        
        # Extended params (backward compatible)
        d_ffn_mult = params.get('d_ffn_mult', 4)
        dropout = params.get('dropout', 0.1)
        weight_decay = params.get('weight_decay', 1e-4)
        pretrain_batch_size = params.get('pretrain_batch_size', 256)
        finetune_patience = params.get('finetune_patience', 20)
        
        try:
            # Build model (with extended params)
            model = CT_CCTL_EMIT(
                n_features=len(self.features),
                n_targets=len(self.targets),
                d_model=d_model,
                n_heads=n_heads,
                n_layers=params['n_layers'],
                d_ffn=d_model * d_ffn_mult,
                dropout=dropout
            )
            
            trainer = CT_CCTL_EMIT_Trainer(
                model, 
                device=self.device, 
                temperature=params['temperature'],
                checkpoint_dir=self.checkpoint_dir
            )
            
            # Pretraining (with extended parameters)
            trainer.pretrain(
                X_source=data['X_source'],
                gvkeys_source=data['gvkeys_source'],
                years_source=data['years_source'],
                epochs=params['pretrain_epochs'],
                lr=params['pretrain_lr'],
                batch_size=pretrain_batch_size,
                weight_decay=weight_decay,
                verbose=verbose
            )
            
            # Fine-tuning (with extended parameters)
            finetune_history = trainer.finetune(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                epochs=params['finetune_epochs'],
                batch_size=params['finetune_batch_size'],
                lr=params['finetune_lr'],
                freeze_encoder=params['freeze_encoder'],
                patience=finetune_patience,
                weight_decay=weight_decay,
                verbose=verbose
            )
            
            # Evaluate train/val/test
            train_results = trainer.evaluate(data['X_train'], data['y_train'])
            val_results = trainer.evaluate(data['X_val'], data['y_val'])
            test_results = trainer.evaluate(data['X_test'], data['y_test'])
            
            train_r2 = train_results.get('R2_Average', -1.0)
            val_r2 = val_results.get('R2_Average', -1.0)
            test_r2 = test_results.get('R2_Average', -1.0)
            
            return {
                'country': country,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'train_results': train_results,
                'val_results': val_results,
                'test_results': test_results,
                'finetune_history': finetune_history,
                'trainer': trainer,
                'success': True
            }
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Training failed for country {country}: {e}")
            return {
                'country': country,
                'train_r2': -1.0,
                'val_r2': -1.0,
                'test_r2': -1.0,
                'success': False,
                'error': str(e)
            }
    
    def _objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective: maximize mean test-set R^2."""
        # Sample hyperparameters (expanded search space)
        params = {
            # Architecture (increased capacity)
            'd_model': trial.suggest_categorical('d_model', [128, 256, 384, 512]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
            'n_layers': trial.suggest_int('n_layers', 3, 6),
            'd_ffn_mult': trial.suggest_categorical('d_ffn_mult', [2, 4]),  # FFN multiplier
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            
            # Pretraining (longer schedule)
            'pretrain_lr': trial.suggest_float('pretrain_lr', 1e-5, 1e-2, log=True),
            'pretrain_epochs': trial.suggest_int('pretrain_epochs', 50, 200, step=25),
            'temperature': trial.suggest_float('temperature', 0.01, 0.2, log=True),
            'pretrain_batch_size': trial.suggest_categorical('pretrain_batch_size', [128, 256, 512]),
            
            # Fine-tuning (more granular)
            'finetune_lr': trial.suggest_float('finetune_lr', 1e-5, 1e-2, log=True),
            'finetune_epochs': trial.suggest_int('finetune_epochs', 100, 300, step=50),
            'finetune_batch_size': trial.suggest_categorical('finetune_batch_size', [16, 32, 64]),
            'finetune_patience': trial.suggest_int('finetune_patience', 20, 50, step=10),
            
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }
        
        # If forcing encoder freeze, do not search this parameter
        if self.force_freeze_encoder:
            params['freeze_encoder'] = True
        else:
            params['freeze_encoder'] = trial.suggest_categorical('freeze_encoder', [True, False])
        
        # Train on all countries
        all_train_r2 = []
        all_val_r2 = []
        all_test_r2 = []
        trial_results = {}
        
        print(f"\nTrial {trial.number}: training on {len(self.target_countries)} countries...")
        
        for country in tqdm(self.target_countries, desc=f"Trial {trial.number}", leave=False):
            result = self._train_single_country(country, params, verbose=False)
            trial_results[country] = result
            
            if result['success']:
                all_train_r2.append(result['train_r2'])
                all_val_r2.append(result['val_r2'])
                all_test_r2.append(result['test_r2'])
        
        # Mean R^2 across splits
        n_success = len(all_test_r2)
        if n_success > 0:
            mean_train_r2 = np.mean(all_train_r2)
            mean_val_r2 = np.mean(all_val_r2)
            mean_test_r2 = np.mean(all_test_r2)
        else:
            mean_train_r2 = mean_val_r2 = mean_test_r2 = -1.0
        
        # Store trial results
        trial.set_user_attr('all_results', trial_results)
        trial.set_user_attr('mean_train_r2', mean_train_r2)
        trial.set_user_attr('mean_val_r2', mean_val_r2)
        trial.set_user_attr('mean_test_r2', mean_test_r2)
        trial.set_user_attr('n_successful', n_success)
        
        print(f"   Trial {trial.number} completed ({n_success}/{len(self.target_countries)} countries succeeded)")
        print(
            f"      Mean train R^2: {mean_train_r2:.4f} | "
            f"Mean val R^2: {mean_val_r2:.4f} | "
            f"Mean test R^2: {mean_test_r2:.4f}"
        )
        
        # Use test-set R^2 as the objective (as requested)
        trial.report(mean_test_r2, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_test_r2  # objective = test-set R^2
    
    def optimize(self) -> Dict[str, Any]:
        """Run multi-country joint hyperparameter optimization."""
        print("="*70)
        print("CT-CCTL-EMIT multi-country joint AutoML search")
        print("   Objective: maximize mean test-set R^2")
        print(f"   #target countries: {len(self.target_countries)}")
        print(f"   Min samples: {self.min_samples}")
        print(f"   Total trials: {self.n_trials}")
        print(f"   Timeout: {self.timeout}s" if self.timeout else "   No timeout")
        print("="*70)
        
        # Create or load the study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best-trial results
        best_trial = study.best_trial
        best_params = study.best_params
        
        # Best-trial details
        best_trial = study.best_trial
        best_train_r2 = best_trial.user_attrs.get('mean_train_r2', -1.0)
        best_val_r2 = best_trial.user_attrs.get('mean_val_r2', -1.0)
        best_test_r2 = study.best_value  # objective is test-set R^2
        
        # Report best result
        print("\n" + "="*70)
        print("Best hyperparameters (maximizing mean test-set R^2):")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        print("\nBest-trial summary:")
        print(f"   Mean train R^2: {best_train_r2:.4f}")
        print(f"   Mean val R^2: {best_val_r2:.4f}")
        print(f"   Mean test R^2: {best_test_r2:.4f} (selection criterion)")
        print("="*70)
        
        # Use best-trial cached results (no retraining needed)
        final_results = best_trial.user_attrs.get('all_results', {})
        
        # If results were not persisted, retrain once with best params
        if not final_results:
            print(f"\nFinal training on {len(self.target_countries)} countries with best params...")
            final_results = {}
            for country in tqdm(self.target_countries, desc="Final training"):
                result = self._train_single_country(country, best_params, verbose=False)
                final_results[country] = result
        
        # Aggregate summary
        all_train_r2 = []
        all_val_r2 = []
        all_test_r2 = []
        for res in final_results.values():
            if res.get('success', False):
                all_train_r2.append(res['train_r2'])
                all_val_r2.append(res['val_r2'])
                all_test_r2.append(res['test_r2'])
        
        summary = {
            'mean_train_r2': np.mean(all_train_r2) if all_train_r2 else -1.0,
            'std_train_r2': np.std(all_train_r2) if all_train_r2 else 0.0,
            'mean_val_r2': np.mean(all_val_r2) if all_val_r2 else -1.0,
            'std_val_r2': np.std(all_val_r2) if all_val_r2 else 0.0,
            'mean_test_r2': np.mean(all_test_r2) if all_test_r2 else -1.0,
            'std_test_r2': np.std(all_test_r2) if all_test_r2 else 0.0,
            'n_successful': len(all_test_r2),
            'n_total': len(self.target_countries),
        }
        
        print("\n" + "="*70)
        print("Final summary:")
        print(f"   Successful countries: {summary['n_successful']}/{summary['n_total']}")
        print(f"   Mean train R^2: {summary['mean_train_r2']:.4f} ± {summary['std_train_r2']:.4f}")
        print(f"   Mean val R^2: {summary['mean_val_r2']:.4f} ± {summary['std_val_r2']:.4f}")
        print(f"   Mean test R^2: {summary['mean_test_r2']:.4f} ± {summary['std_test_r2']:.4f}")
        print("="*70)
        
        return {
            'best_params': best_params,
            'best_mean_val_r2': study.best_value,
            'summary': summary,
            'all_country_results': final_results,
            'study': study,
            'target_countries': self.target_countries,
        }
    
    def get_country_results_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        rows = []
        for country, res in results['all_country_results'].items():
            if res.get('success', False):
                row = {
                    'country': country,
                    'train_r2': res.get('train_r2', np.nan),
                    'val_r2': res.get('val_r2', np.nan),
                    'test_r2': res.get('test_r2', np.nan),
                    'test_mae': res.get('test_results', {}).get('MAE_Average', np.nan),
                    'test_mse': res.get('test_results', {}).get('MSE_Average', np.nan),
                }
                # Add per-target R^2
                test_results = res.get('test_results', {})
                for key, val in test_results.items():
                    if key.startswith('R2_') and key != 'R2_Average':
                        target_name = key.replace('R2_', '')
                        row[f'test_r2_{target_name}'] = val
                rows.append(row)
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('test_r2', ascending=False).reset_index(drop=True)
        return df


def run_multi_country_automl(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    target_countries: Optional[List[str]] = None,
    min_samples: int = 50,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    device: str = 'cuda',
    random_state: int = 42,
    checkpoint_dir: Optional[str] = None,
    force_freeze_encoder: bool = False,
    save_trial_results: bool = True,  # whether to persist per-trial results
) -> Dict[str, Any]:
    """
    Convenience wrapper for multi-country joint AutoML (Plan A).

    Args:
        df: full dataset
        features: feature column names
        targets: target column names
        target_countries: target countries (if None, auto-select by min_samples)
        min_samples: minimum effective sample size (default: 50)
        n_trials: number of trials
        timeout: timeout (seconds)
        device: device
        random_state: random seed
        checkpoint_dir: checkpoint directory
        force_freeze_encoder: force freezing the encoder (recommended to mitigate catastrophic forgetting)
        save_trial_results: persist each trial to SQLite (supports resume)

    Returns:
        dict with best params and per-country test results
    """
    if target_countries is None:
        # Auto-select eligible countries
        all_countries = df['loc'].unique().tolist()
        target_countries = []
        for country in all_countries:
            df_country = df[df['loc'] == country]
            valid_mask = df_country[targets].notna().all(axis=1)
            if valid_mask.sum() >= min_samples:
                target_countries.append(country)
        print(f"🔍 Auto-selected eligible countries: {len(target_countries)}")
    
    # Optuna storage (SQLite) for resume/recovery
    storage = None
    if save_trial_results and checkpoint_dir:
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        db_path = os.path.join(checkpoint_dir, 'optuna_study.db')
        storage = f'sqlite:///{db_path}'
        print(f"💾 Optuna storage: {db_path}")
        print("   ✅ Auto-saved after each trial (resume supported)")
    
    automl = CT_CCTL_EMIT_MultiCountryAutoML(
        df=df,
        features=features,
        targets=targets,
        target_countries=target_countries,
        min_samples=min_samples,
        device=device,
        random_state=random_state,
        n_trials=n_trials,
        timeout=timeout,
        storage=storage,  # pass storage URI
        checkpoint_dir=checkpoint_dir,
        force_freeze_encoder=force_freeze_encoder,
    )
    
    return automl.optimize()


# ==============================================================================
# 11. Multi-GPU parallel acceleration
# ==============================================================================

def _train_country_on_gpu(args):
    """
    Train a single country on a specified GPU (for multi-process parallelism).

    Note: this runs in a child process and thus re-creates the model.
    """
    country, country_data, params, features, targets, gpu_id, checkpoint_dir = args
    
    device = f'cuda:{gpu_id}'
    
    # Ensure d_model is divisible by n_heads
    d_model = params['d_model']
    n_heads = params['n_heads']
    if d_model % n_heads != 0:
        d_model = (d_model // n_heads) * n_heads
    
    try:
        # Build model
        model = CT_CCTL_EMIT(
            n_features=len(features),
            n_targets=len(targets),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=params['n_layers']
        )
        
        trainer = CT_CCTL_EMIT_Trainer(
            model, 
            device=device, 
            temperature=params['temperature'],
            checkpoint_dir=checkpoint_dir
        )
        
        # Pretraining
        trainer.pretrain(
            X_source=country_data['X_source'],
            gvkeys_source=country_data['gvkeys_source'],
            years_source=country_data['years_source'],
            epochs=params['pretrain_epochs'],
            lr=params['pretrain_lr'],
            verbose=False
        )
        
        # Fine-tuning
        finetune_history = trainer.finetune(
            X_train=country_data['X_train'],
            y_train=country_data['y_train'],
            X_val=country_data['X_val'],
            y_val=country_data['y_val'],
            epochs=params['finetune_epochs'],
            batch_size=params['finetune_batch_size'],
            lr=params['finetune_lr'],
            freeze_encoder=params['freeze_encoder'],
            verbose=False
        )
        
        # Evaluation
        val_results = trainer.evaluate(country_data['X_val'], country_data['y_val'])
        test_results = trainer.evaluate(country_data['X_test'], country_data['y_test'])
        
        val_r2 = val_results.get('R2_Average', -1.0)
        
        return {
            'country': country,
            'val_r2': val_r2,
            'val_results': val_results,
            'test_results': test_results,
            'success': True
        }
        
    except Exception as e:
        return {
            'country': country,
            'val_r2': -1.0,
            'success': False,
            'error': str(e)
        }


class CT_CCTL_EMIT_MultiGPU_AutoML:
    """
    Multi-GPU parallel hyperparameter optimization.

    Acceleration strategy:
    1. Country-level parallelism: distribute countries across GPUs.
    2. Optuna-style parallel workers for different trials.

    Example (4 GPUs):
    - Per trial: 55 countries / 4 GPUs ≈ 14 countries per GPU (in parallel)
    - Theoretical speedup: 4x (typically ~3–3.5x due to overhead)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        targets: List[str],
        target_countries: List[str],
        gpu_ids: List[int] = None,
        min_samples: int = 50,
        random_state: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        n_workers_per_gpu: int = 1,
    ):
        """
        Args:
            df: full dataset
            features: feature column names
            targets: target column names
            target_countries: target country list
            gpu_ids: GPU IDs (e.g., [0, 1, 2, 3]); auto-detect if None
            min_samples: minimum effective sample size
            random_state: random seed
            n_trials: number of trials
            timeout: timeout (seconds)
            study_name: study name
            storage: Optuna DB path (for distributed runs)
            checkpoint_dir: checkpoint directory
            n_workers_per_gpu: workers per GPU (for small models)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Run: pip install optuna")
        
        self.df = df
        self.features = features
        self.targets = targets
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or 'cctl_emit_multi_gpu'
        self.storage = storage
        self.checkpoint_dir = checkpoint_dir
        self.min_samples = min_samples
        self.n_workers_per_gpu = n_workers_per_gpu
        
        # Detect available GPUs
        if gpu_ids is None:
            n_gpus = torch.cuda.device_count()
            self.gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [0]
        else:
            self.gpu_ids = gpu_ids
        
        print(f"🖥️ Detected {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        
        # Filter eligible countries
        self.target_countries = self._filter_eligible_countries(target_countries)
        print(f"✅ Eligible countries: {len(self.target_countries)}")
        
        # Store per-country data
        self.country_data = {}
        
        # Preprocess data
        self._prepare_all_data()
    
    def _filter_eligible_countries(self, countries: List[str]) -> List[str]:
        """Filter countries with effective sample size >= min_samples."""
        eligible = []
        for country in countries:
            df_country = self.df[self.df['loc'] == country]
            valid_mask = df_country[self.targets].notna().all(axis=1)
            n_valid = valid_mask.sum()
            if n_valid >= self.min_samples:
                eligible.append(country)
        return eligible
    
    def _prepare_all_data(self):
        """Preprocess data for all countries."""
        print("\n📊 Preprocessing data for all countries...")
        
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Global scaler
        self.global_scaler = StandardScaler()
        self.global_scaler.fit(self.df[self.features].values)
        
        for country in tqdm(self.target_countries, desc="Preparing data"):
            source_countries = [c for c in self.df['loc'].unique() if c != country]
            
            df_source = self.df[self.df['loc'].isin(source_countries)].copy()
            df_target = self.df[self.df['loc'] == country].copy()
            
            X_source = self.global_scaler.transform(df_source[self.features].values)
            X_target = self.global_scaler.transform(df_target[self.features].values)
            
            y_source = df_source[self.targets].values
            y_target = df_target[self.targets].values
            
            gvkeys_source = df_source['gvkey'].values
            years_source = df_source['fiscalyear'].values
            
            years_target = df_target['fiscalyear'].values
            unique_years = sorted(df_target['fiscalyear'].unique())
            
            if len(unique_years) >= 3:
                train_years = unique_years[:-2]
                val_years = [unique_years[-2]]
                test_years = [unique_years[-1]]
                
                train_mask = np.isin(years_target, train_years)
                val_mask = np.isin(years_target, val_years)
                test_mask = np.isin(years_target, test_years)
                
                X_train, y_train = X_target[train_mask], y_target[train_mask]
                X_val, y_val = X_target[val_mask], y_target[val_mask]
                X_test, y_test = X_target[test_mask], y_target[test_mask]
            else:
                indices = list(range(len(X_target)))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.3, random_state=self.random_state
                )
                val_idx = test_idx[:len(test_idx)//2]
                test_idx = test_idx[len(test_idx)//2:]
                
                X_train = X_target[train_idx]
                X_val = X_target[val_idx]
                X_test = X_target[test_idx]
                y_train = y_target[train_idx]
                y_val = y_target[val_idx]
                y_test = y_target[test_idx]
            
            self.country_data[country] = {
                'X_source': X_source,
                'y_source': y_source,
                'gvkeys_source': gvkeys_source,
                'years_source': years_source,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
            }
        
        print(f"✅ Preprocessing complete: {len(self.country_data)} countries")
    
    def _train_countries_parallel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all countries in parallel across multiple GPUs.

        Strategy: assign countries to GPUs (round-robin) and run with
        ThreadPoolExecutor.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        n_gpus = len(self.gpu_ids)
        countries = list(self.target_countries)
        
        # Assign countries to GPUs (round-robin for load balancing)
        country_gpu_assignments = []
        for i, country in enumerate(countries):
            gpu_id = self.gpu_ids[i % n_gpus]
            country_gpu_assignments.append((country, gpu_id))
        
        all_results = {}
        
        def train_on_gpu(country: str, gpu_id: int) -> Dict:
            """Train on a specified GPU."""
            device = f'cuda:{gpu_id}'
            data = self.country_data[country]
            
            d_model = params['d_model']
            n_heads = params['n_heads']
            if d_model % n_heads != 0:
                d_model = (d_model // n_heads) * n_heads
            
            try:
                model = CT_CCTL_EMIT(
                    n_features=len(self.features),
                    n_targets=len(self.targets),
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=params['n_layers']
                )
                
                trainer = CT_CCTL_EMIT_Trainer(
                    model, 
                    device=device, 
                    temperature=params['temperature'],
                    checkpoint_dir=None  # no checkpoints in parallel mode
                )
                
                trainer.pretrain(
                    X_source=data['X_source'],
                    gvkeys_source=data['gvkeys_source'],
                    years_source=data['years_source'],
                    epochs=params['pretrain_epochs'],
                    lr=params['pretrain_lr'],
                    verbose=False
                )
                
                finetune_history = trainer.finetune(
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_val=data['X_val'],
                    y_val=data['y_val'],
                    epochs=params['finetune_epochs'],
                    batch_size=params['finetune_batch_size'],
                    lr=params['finetune_lr'],
                    freeze_encoder=params['freeze_encoder'],
                    verbose=False
                )
                
                # Evaluate train/val/test
                train_results = trainer.evaluate(data['X_train'], data['y_train'])
                val_results = trainer.evaluate(data['X_val'], data['y_val'])
                test_results = trainer.evaluate(data['X_test'], data['y_test'])
                
                return {
                    'country': country,
                    'train_r2': train_results.get('R2_Average', -1.0),
                    'val_r2': val_results.get('R2_Average', -1.0),
                    'test_r2': test_results.get('R2_Average', -1.0),
                    'train_results': train_results,
                    'val_results': val_results,
                    'test_results': test_results,
                    'success': True
                }
                
            except Exception as e:
                return {
                    'country': country,
                    'train_r2': -1.0,
                    'val_r2': -1.0,
                    'test_r2': -1.0,
                    'success': False,
                    'error': str(e)
                }
        
        # Parallel execution via thread pool (GPU ops release the GIL)
        max_workers = n_gpus * self.n_workers_per_gpu
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(train_on_gpu, country, gpu_id): country 
                for country, gpu_id in country_gpu_assignments
            }
            
            for future in as_completed(futures):
                result = future.result()
                all_results[result['country']] = result
        
        return all_results
    
    def _objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective (multi-GPU): maximize mean test-set R^2."""
        params = {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 192, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'pretrain_lr': trial.suggest_float('pretrain_lr', 1e-4, 1e-2, log=True),
            'pretrain_epochs': trial.suggest_int('pretrain_epochs', 20, 80, step=10),
            'temperature': trial.suggest_float('temperature', 0.03, 0.2, log=True),
            'finetune_lr': trial.suggest_float('finetune_lr', 1e-4, 1e-2, log=True),
            'finetune_epochs': trial.suggest_int('finetune_epochs', 50, 150, step=25),
            'finetune_batch_size': trial.suggest_categorical('finetune_batch_size', [32, 64, 128]),
            'freeze_encoder': trial.suggest_categorical('freeze_encoder', [True, False]),
        }
        
        print(
            f"\n🔄 Trial {trial.number}: parallel training on "
            f"{len(self.target_countries)} countries across {len(self.gpu_ids)} GPUs..."
        )
        
        # Multi-GPU parallel training
        trial_results = self._train_countries_parallel(params)
        
        # Mean R^2 across splits
        all_train_r2 = [r['train_r2'] for r in trial_results.values() if r['success']]
        all_val_r2 = [r['val_r2'] for r in trial_results.values() if r['success']]
        all_test_r2 = [r['test_r2'] for r in trial_results.values() if r['success']]
        
        n_success = len(all_test_r2)
        mean_train_r2 = np.mean(all_train_r2) if all_train_r2 else -1.0
        mean_val_r2 = np.mean(all_val_r2) if all_val_r2 else -1.0
        mean_test_r2 = np.mean(all_test_r2) if all_test_r2 else -1.0
        
        trial.set_user_attr('all_results', trial_results)
        trial.set_user_attr('mean_train_r2', mean_train_r2)
        trial.set_user_attr('mean_val_r2', mean_val_r2)
        trial.set_user_attr('mean_test_r2', mean_test_r2)
        trial.set_user_attr('n_successful', n_success)
        
        print(f"   ✅ Trial {trial.number} completed ({n_success}/{len(self.target_countries)} countries succeeded)")
        print(
            f"      Mean train R^2: {mean_train_r2:.4f} | "
            f"Mean val R^2: {mean_val_r2:.4f} | "
            f"Mean test R^2: {mean_test_r2:.4f}"
        )
        
        # Use test-set R^2 as objective
        trial.report(mean_test_r2, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_test_r2  # objective = test-set R^2
    
    def optimize(self) -> Dict[str, Any]:
        """Run multi-GPU hyperparameter optimization."""
        print("="*70)
        print("🚀 CT-CCTL-EMIT multi-GPU AutoML search")
        print("   Objective: maximize mean test-set R^2")
        print(f"   #GPUs: {len(self.gpu_ids)}")
        print(f"   GPU IDs: {self.gpu_ids}")
        print(f"   #target countries: {len(self.target_countries)}")
        print(f"   Workers per GPU: {self.n_workers_per_gpu}")
        print(f"   Total trials: {self.n_trials}")
        print(f"   Theoretical speedup: ~{len(self.gpu_ids)}x")
        print("="*70)
        
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        
        # Retrieve detailed information for the best trial
        best_trial = study.best_trial
        best_train_r2 = best_trial.user_attrs.get('mean_train_r2', -1.0)
        best_val_r2 = best_trial.user_attrs.get('mean_val_r2', -1.0)
        best_test_r2 = study.best_value  # Optimization target: mean test-set R²
        
        print("\n" + "="*70)
        print("Best hyperparameters (maximizing mean test-set R²):")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        print("\nBest-trial summary:")
        print(f"   Mean train R²: {best_train_r2:.4f}")
        print(f"   Mean validation R²: {best_val_r2:.4f}")
        print(f"   Mean test R²: {best_test_r2:.4f}  (selection criterion)")
        print("="*70)
        
        # Use the best trial's cached results if available
        final_results = best_trial.user_attrs.get('all_results', {})
        
        # If results were not cached, run final training with the best hyperparameters
        if not final_results:
            print("\nRunning final training with the best hyperparameters...")
            final_results = self._train_countries_parallel(best_params)
        
        # Summary statistics
        all_train_r2 = []
        all_val_r2 = []
        all_test_r2 = []
        for res in final_results.values():
            if res.get('success', False):
                all_train_r2.append(res['train_r2'])
                all_val_r2.append(res['val_r2'])
                all_test_r2.append(res['test_r2'])
        
        summary = {
            'mean_train_r2': np.mean(all_train_r2) if all_train_r2 else -1.0,
            'std_train_r2': np.std(all_train_r2) if all_train_r2 else 0.0,
            'mean_val_r2': np.mean(all_val_r2) if all_val_r2 else -1.0,
            'std_val_r2': np.std(all_val_r2) if all_val_r2 else 0.0,
            'mean_test_r2': np.mean(all_test_r2) if all_test_r2 else -1.0,
            'std_test_r2': np.std(all_test_r2) if all_test_r2 else 0.0,
            'n_successful': len(all_test_r2),
            'n_total': len(self.target_countries),
            'n_gpus': len(self.gpu_ids),
        }
        
        print("\n" + "="*70)
        print("Final summary:")
        print(f"   Successful countries: {summary['n_successful']}/{summary['n_total']}")
        print(f"   Mean train R²: {summary['mean_train_r2']:.4f} ± {summary['std_train_r2']:.4f}")
        print(f"   Mean validation R²: {summary['mean_val_r2']:.4f} ± {summary['std_val_r2']:.4f}")
        print(f"   Mean test R²: {summary['mean_test_r2']:.4f} ± {summary['std_test_r2']:.4f}")
        print("="*70)
        
        return {
            'best_params': best_params,
            'best_mean_train_r2': best_train_r2,
            'best_mean_val_r2': best_val_r2,
            'best_mean_test_r2': best_test_r2,
            'summary': summary,
            'all_country_results': final_results,
            'study': study,
            'target_countries': self.target_countries,
        }


def run_multi_gpu_automl(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    target_countries: Optional[List[str]] = None,
    gpu_ids: Optional[List[int]] = None,
    min_samples: int = 50,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    random_state: int = 42,
    checkpoint_dir: Optional[str] = None,
    n_workers_per_gpu: int = 1,
) -> Dict[str, Any]:
    """
    Convenience wrapper for multi-GPU parallel AutoML.
    
    Args:
        df: full dataset
        features: feature column names
        targets: target column names
        target_countries: list of target countries
        gpu_ids: GPU ID list (e.g., [0, 1, 2, 3]); auto-detected if None
        min_samples: minimum effective sample size per country
        n_trials: number of optimization trials
        timeout: timeout in seconds
        random_state: random seed
        checkpoint_dir: checkpoint directory
        n_workers_per_gpu: number of concurrent workers per GPU
        
    Returns:
        A dictionary containing the best hyperparameters and per-country test results.
    
    Example:
        # Use all available GPUs
        result = run_multi_gpu_automl(df, features, targets)
        
        # Use GPU 0 and 1 only
        result = run_multi_gpu_automl(df, features, targets, gpu_ids=[0, 1])
    """
    if target_countries is None:
        all_countries = df['loc'].unique().tolist()
        target_countries = []
        for country in all_countries:
            df_country = df[df['loc'] == country]
            valid_mask = df_country[targets].notna().all(axis=1)
            if valid_mask.sum() >= min_samples:
                target_countries.append(country)
        print(f"Auto-selected eligible countries: {len(target_countries)}")
    
    automl = CT_CCTL_EMIT_MultiGPU_AutoML(
        df=df,
        features=features,
        targets=targets,
        target_countries=target_countries,
        gpu_ids=gpu_ids,
        min_samples=min_samples,
        random_state=random_state,
        n_trials=n_trials,
        timeout=timeout,
        checkpoint_dir=checkpoint_dir,
        n_workers_per_gpu=n_workers_per_gpu,
    )
    
    return automl.optimize()
