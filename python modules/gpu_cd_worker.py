import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd
import torch

from model_trainer_all import GlobalExperimentRunner, run_global_experiments


def load_data_raw(project_root_path: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    data_dir_local = os.path.join(project_root_path, 'data')
    processed_hdf5_path = os.path.join(data_dir_local, 'processed_feature_data.h5')

    target_cols = [
        'log_Scope1', 'log_Scope2', 'log_Scope3_upstream',
        'log_Scope3_downLA', 'log_Scope3_prod', 'log_Scope_total',
    ]

    possible_keys = ['processed_data', 'processed_feature_data', 'data', 'df_processed', '/processed_features']
    df_local = None
    with pd.HDFStore(processed_hdf5_path, 'r') as store:
        available_keys = list(store.keys())
        for key in possible_keys:
            if key in available_keys:
                df_local = pd.read_hdf(processed_hdf5_path, key=key)
                break
        if df_local is None:
            df_local = pd.read_hdf(processed_hdf5_path, key=available_keys[0])

    valid_loc = (
        df_local['loc'].notna() & (df_local['loc'] != '') & (df_local['loc'].astype(str).str.strip() != '')
        & (df_local['loc'].astype(str).str.len() >= 2)
        & (df_local['loc'].astype(str).str.isalpha())
        & (df_local['loc'].astype(str).str.len() <= 10)
    )
    df_local = df_local[valid_loc].copy()

    id_cols = ['gvkey', 'fiscalyear', 'loc', 'GICSSector']
    exclude_cols = list(target_cols) + id_cols
    original_targets = ['Scope1', 'Scope2', 'Scope3_upstream', 'Scope3_prod', 'Scope3_downLA', 'Scope_total']
    exclude_cols.extend([c for c in original_targets if c in df_local.columns])
    other_exclude = [c for c in df_local.columns if 'sector' in c.lower() and c not in id_cols]
    exclude_cols.extend(other_exclude)
    feature_cols = [c for c in df_local.columns if c not in exclude_cols]

    df_labeled_local = df_local[df_local[target_cols].notnull().all(axis=1)].copy()

    developed_countries = [
        'USA', 'CAN', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'CHE', 'AUT', 'IRL', 'LUX', 'PRT',
        'SWE', 'NOR', 'DNK', 'FIN', 'GRC', 'POL', 'JPN', 'KOR', 'TWN', 'HKG', 'SGP',
        'ISR', 'SAU', 'ARE', 'KWT', 'QAT', 'AUS', 'NZL', 'CHL', 'BMU', 'CYM',
    ]
    _loc_set = set(df_local['loc'].unique())
    developed_countries = [c for c in developed_countries if c in _loc_set]

    return df_labeled_local, feature_cols, target_cols, developed_countries


def worker_run_fixed(
    result_q,
    worker_id: int,
    gpu_id: int,
    run_id: str,
    project_root_path: str,
    results_dir_path: str,
    log_dir_path: str,
    seed_countries,
    experiments_list,
    quick_run: bool,
    loader_cfg: Dict[str, object],
    batch_cfg: Dict[str, int],
    tune_cfg: Dict[str, int],
) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)

    print(f"🧩 Worker {worker_id} start | GPU {gpu_id}")
    print(f"🔧 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"🔧 torch.cuda.device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device()}")
    print(f"🔧 QUICK_RUN={quick_run}")
    print(f"🔧 DataLoader config={loader_cfg}")
    print(f"🔧 Batch config={batch_cfg}")
    print(f"🔧 Tune config={tune_cfg}")

    df_labeled_local, feature_cols, target_cols, developed_countries = load_data_raw(project_root_path)

    shard_rows = []
    for seed, countries in seed_countries:
        print(f"{'='*70}")
        print(f"Worker {worker_id} | Seed = {seed}")

        runner = GlobalExperimentRunner(
            df_labeled_local,
            feature_cols,
            target_cols,
            random_state=seed,
            developed_countries=developed_countries,
            split_mode='time',
            dataloader_num_workers=loader_cfg['num_workers'],
            dataloader_pin_memory=loader_cfg['pin_memory'],
            dataloader_persistent_workers=loader_cfg['persistent_workers'],
            dataloader_prefetch_factor=loader_cfg['prefetch_factor'],
        )

        for country in countries:
            for exp in experiments_list:
                rows = run_global_experiments(
                    df_labeled_local,
                    feature_cols,
                    target_cols,
                    countries=[country],
                    min_samples=50,
                    experiments=[exp],
                    runner=runner,
                    random_state=seed,
                    developed_countries=developed_countries,
                    checkpoint_dir=os.path.join(results_dir_path, 'checkpoints'),
                    checkpoint_run_id=f"gpu_cd_seed_{seed}",  # fixed ID to reuse existing checkpoints
                    resume=True,
                    quick_run=quick_run,
                    dataloader_num_workers=loader_cfg['num_workers'],
                    dataloader_pin_memory=loader_cfg['pin_memory'],
                    dataloader_persistent_workers=loader_cfg['persistent_workers'],
                    dataloader_prefetch_factor=loader_cfg['prefetch_factor'],
                    n_finetune_trials=tune_cfg['n_finetune_trials'],
                    pretrain_epochs=tune_cfg['mlp_pretrain_epochs'],
                    finetune_epochs=tune_cfg['mlp_finetune_epochs'],
                    ft_pretrain_epochs=tune_cfg['ft_pretrain_epochs'],
                    ft_finetune_epochs=tune_cfg['ft_finetune_epochs'],
                    lora_rank=tune_cfg['lora_rank'],
                    lora_alpha=tune_cfg['lora_alpha'],
                    mlp_pretrain_batch_size=batch_cfg['mlp_pretrain'],
                    mlp_finetune_batch_size=batch_cfg['mlp_finetune'],
                    mlp_eval_batch_size=batch_cfg['mlp_eval'],
                    ft_pretrain_batch_size=batch_cfg['ft_pretrain'],
                    ft_finetune_batch_size=batch_cfg['ft_finetune'],
                )
                for r in rows:
                    r = dict(r)
                    r['seed'] = seed
                    shard_rows.append(r)
                result_q.put({'type': 'progress', 'count': 1})

    shard_pkl = os.path.join(results_dir_path, f'gpu_cd_{run_id}_worker{worker_id}.pkl')
    with open(shard_pkl, 'wb') as pf:
        pickle.dump(shard_rows, pf)

    result_q.put({'type': 'done', 'worker_id': worker_id, 'pkl': shard_pkl, 'log': None})
