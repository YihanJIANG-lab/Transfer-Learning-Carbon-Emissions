import os
import json
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ct_cctl_emit import run_ct_cctl_emit_experiment


def load_ct_data(project_root_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    data_dir = os.path.join(project_root_path, 'data')
    processed_hdf5_path = os.path.join(data_dir, 'processed_feature_data.h5')

    target_cols = [
        'log_Scope1', 'log_Scope2', 'log_Scope3_upstream',
        'log_Scope3_downLA', 'log_Scope3_prod', 'log_Scope_total',
    ]

    possible_keys = ['processed_data', 'processed_feature_data', 'data', 'df_processed', '/processed_features']
    df = None
    with pd.HDFStore(processed_hdf5_path, 'r') as store:
        available_keys = list(store.keys())
        for key in possible_keys:
            if key in available_keys:
                df = pd.read_hdf(processed_hdf5_path, key=key)
                break
        if df is None:
            df = pd.read_hdf(processed_hdf5_path, key=available_keys[0])

    valid_loc = (
        df['loc'].notna() & (df['loc'] != '') & (df['loc'].astype(str).str.strip() != '')
        & (df['loc'].astype(str).str.len() >= 2)
        & (df['loc'].astype(str).str.isalpha())
        & (df['loc'].astype(str).str.len() <= 10)
    )
    df = df[valid_loc].copy()

    id_cols = ['gvkey', 'fiscalyear', 'loc', 'GICSSector']
    exclude_cols = list(target_cols) + id_cols
    original_targets = ['Scope1', 'Scope2', 'Scope3_upstream', 'Scope3_prod', 'Scope3_downLA', 'Scope_total']
    exclude_cols.extend([c for c in original_targets if c in df.columns])
    other_exclude = [c for c in df.columns if 'sector' in c.lower() and c not in id_cols]
    exclude_cols.extend(other_exclude)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df_labeled = df[df[target_cols].notnull().all(axis=1)].copy()
    return df, df_labeled, feature_cols, target_cols


def ct_worker_run(
    result_q,
    worker_id: int,
    gpu_id: int,
    project_root: str,
    results_dir: str,
    countries: List[str],
    seeds: List[int],
    run_strategies: List[str],
    strategy_labels: Dict[str, str],
    ct_full_cfg: Dict[str, object],
    ct_lora_cfg: Dict[str, object],
) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)

    df, df_labeled, feature_cols, target_cols = load_ct_data(project_root)

    checkpoint_roots = {
        'full': os.path.join(results_dir, 'checkpoints', 'c_ct_v3'),
        'lora': os.path.join(results_dir, 'checkpoints', 'd_lora_ct_v3'),
    }
    for _, root in checkpoint_roots.items():
        os.makedirs(root, exist_ok=True)

    def _to_py(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_py(v) for v in obj]
        return obj

    def _result_paths(ckpt_dir):
        return (
            os.path.join(ckpt_dir, 'result.pkl'),
            os.path.join(ckpt_dir, 'result.json'),
        )

    def _load_cached_result(ckpt_dir):
        pkl_path, json_path = _result_paths(ckpt_dir)
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return None

    def _save_cached_result(ckpt_dir, res):
        pkl_path, json_path = _result_paths(ckpt_dir)
        with open(pkl_path, 'wb') as f:
            pickle.dump(res, f)
        with open(json_path, 'w') as f:
            json.dump(_to_py(res), f, ensure_ascii=False, indent=2)

    rows = []
    for strategy in run_strategies:
        cfg = ct_full_cfg if strategy == 'full' else ct_lora_cfg
        strategy_label = strategy_labels[strategy]
        checkpoint_root = checkpoint_roots[strategy]

        for seed in seeds:
            for country in countries:
                if strategy == 'lora':
                    tag = f"{country}_seed={seed}_t={cfg['temperature']}_r={cfg['lora_rank']}"
                else:
                    tag = f"{country}_seed={seed}_t={cfg['temperature']}_full"
                ckpt_dir = os.path.join(checkpoint_root, tag)
                os.makedirs(ckpt_dir, exist_ok=True)

                cached = _load_cached_result(ckpt_dir)
                if cached is not None:
                    rows.append(cached)
                    result_q.put({'type': 'progress', 'count': 1})
                    continue

                # CRITICAL: freeze_encoder controls whether to freeze encoder during fine-tuning
                # C_CT (full): freeze_encoder=False → train entire model
                # D_lora_CT (lora): freeze_encoder=True → train only LoRA + head
                freeze_encoder = cfg.get('freeze_encoder', strategy == 'lora')
                
                res = run_ct_cctl_emit_experiment(
                    df=df_labeled,
                    df_pretrain=df,
                    df_finetune=df_labeled,
                    features=feature_cols,
                    targets=target_cols,
                    target_country=country,
                    pretrain_epochs=cfg['pretrain_epochs'],
                    finetune_epochs=cfg['finetune_epochs'],
                    pretrain_lr=cfg.get('pretrain_lr', 1e-3),
                    finetune_lr=cfg.get('finetune_lr', 1e-3),
                    finetune_patience=cfg.get('finetune_patience', 20),
                    finetune_weight_decay=cfg.get('finetune_weight_decay', 1e-4),
                    pretrain_batch_size=cfg['pretrain_batch_size'],
                    finetune_batch_size=cfg['finetune_batch_size'],
                    d_model=cfg['d_model'],
                    n_heads=cfg['n_heads'],
                    n_layers=cfg['n_layers'],
                    temperature=cfg['temperature'],
                    freeze_encoder=freeze_encoder,  # CRITICAL: pass from config
                    finetune_strategy=('lora' if strategy == 'lora' else 'full'),
                    lora_rank=cfg['lora_rank'],
                    lora_alpha=cfg['lora_alpha'],
                    random_state=seed,
                    device=cfg['device'],
                    checkpoint_dir=ckpt_dir,
                    verbose=True,
                    ensemble_strategy=cfg.get('ensemble_strategy', 'ridge_calibration'),
                    mlp_hidden_sizes=tuple(cfg.get('mlp_hidden_sizes', (512, 256))),
                    mlp_dropout=cfg.get('mlp_dropout', 0.2),
                    mlp_epochs=cfg.get('mlp_epochs', 200),
                    mlp_lr=cfg.get('mlp_lr', 1e-3),
                    mlp_batch_size=cfg.get('mlp_batch_size', 256),
                    mlp_patience=cfg.get('mlp_patience', 20),
                    mlp_weight_decay=cfg.get('mlp_weight_decay', 1e-4),
                    dataloader_num_workers=cfg['loader_num_workers'],
                    dataloader_pin_memory=cfg['loader_pin_memory'],
                    dataloader_persistent_workers=cfg['loader_persistent_workers'],
                    dataloader_prefetch_factor=cfg['loader_prefetch_factor'],
                    use_amp=cfg.get('use_amp', False),
                    pretrain_checkpoint_path=cfg.get('pretrain_checkpoint_path'),
                    skip_pretrain=cfg.get('skip_pretrain', False),
                )
                res = dict(res)
                res.update({'seed': seed, 'country': country, 'experiment_code': strategy_label})
                _save_cached_result(ckpt_dir, res)
                rows.append(res)
                result_q.put({'type': 'progress', 'count': 1})

    shard_pkl = os.path.join(results_dir, f'ct_worker_{worker_id}.pkl')
    with open(shard_pkl, 'wb') as pf:
        pickle.dump(rows, pf)

    result_q.put({'type': 'done', 'worker_id': worker_id, 'pkl': shard_pkl})
