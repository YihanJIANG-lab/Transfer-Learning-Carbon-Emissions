import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class CheckpointRunInfo:
    signature: str
    path: str
    n_files: int
    mtime: float


def _safe_load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _is_result_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and ('country' in obj or 'model' in obj or 'experiment_code' in obj)


def _is_wrapper(obj: Any) -> bool:
    return isinstance(obj, dict) and 'result' in obj and 'signature' in obj


def _iter_checkpoint_files(checkpoint_root: str) -> List[str]:
    if not os.path.isdir(checkpoint_root):
        return []
    files: List[str] = []
    for name in os.listdir(checkpoint_root):
        if not name.endswith('.pkl'):
            continue
        if name.startswith('.tmp_'):
            continue
        files.append(os.path.join(checkpoint_root, name))
    return sorted(files)


def list_checkpoint_runs(checkpoint_base_dir: str) -> List[CheckpointRunInfo]:
    """List available checkpoint runs under a base directory.

    Expected structure:
      checkpoint_base_dir/<signature>/*.pkl

    Returns sorted by mtime desc.
    """
    if not os.path.isdir(checkpoint_base_dir):
        return []

    runs: List[CheckpointRunInfo] = []
    for name in os.listdir(checkpoint_base_dir):
        run_path = os.path.join(checkpoint_base_dir, name)
        if not os.path.isdir(run_path):
            continue
        files = _iter_checkpoint_files(run_path)
        if not files:
            continue
        try:
            mtime = max(os.path.getmtime(p) for p in files)
        except Exception:
            mtime = os.path.getmtime(run_path)
        runs.append(CheckpointRunInfo(signature=name, path=run_path, n_files=len(files), mtime=mtime))

    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def pick_checkpoint_run(checkpoint_base_dir: str, signature: Optional[str] = None) -> CheckpointRunInfo:
    runs = list_checkpoint_runs(checkpoint_base_dir)
    if not runs:
        raise FileNotFoundError(f"No checkpoint runs found under: {checkpoint_base_dir}")

    if signature is None or signature in ('latest', 'newest'):
        return runs[0]

    for run in runs:
        if run.signature == signature:
            return run

    raise FileNotFoundError(
        f"Signature '{signature}' not found under {checkpoint_base_dir}. Available: {[r.signature for r in runs]}"
    )


def load_checkpoint_results(
    checkpoint_run_dir: str,
    signature: Optional[str] = None,
    include_failed: bool = True,
    strict_signature: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load checkpoint results from a run directory.

    Args:
      checkpoint_run_dir: directory like .../<signature>/ containing per (country, exp) pkl files.
      signature: if provided, verify wrapper signature matches.
      include_failed: include results with status or missing/NaN R2.
      strict_signature: when signature mismatch, skip (True) or allow (False).

    Returns: (results, skipped_files)
    """
    results: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for path in _iter_checkpoint_files(checkpoint_run_dir):
        try:
            obj = _safe_load_pickle(path)
        except Exception:
            skipped.append(path)
            continue

        if _is_wrapper(obj):
            if signature is not None and obj.get('signature') != signature:
                if strict_signature:
                    skipped.append(path)
                    continue
            res = obj.get('result')
            if not _is_result_dict(res):
                skipped.append(path)
                continue
            results.append(res)
            continue

        if _is_result_dict(obj):
            # legacy / direct result dict
            results.append(obj)
            continue

        skipped.append(path)

    if not include_failed:
        cleaned: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict) and ('status' not in r):
                cleaned.append(r)
        results = cleaned

    return results, skipped


def merge_checkpoints(
    checkpoint_base_dir: str,
    signature: Optional[str] = None,
    out_pkl: Optional[str] = None,
    out_xlsx: Optional[str] = None,
    include_failed: bool = True,
) -> Dict[str, Any]:
    """Merge a checkpoint run into consolidated outputs.

    Args:
      checkpoint_base_dir: base dir containing run subdirs; e.g. results/checkpoints
      signature: run signature or 'latest'
      out_pkl/out_xlsx: output paths (optional)

    Returns summary dict.
    """
    run = pick_checkpoint_run(checkpoint_base_dir, signature=signature)
    results, skipped = load_checkpoint_results(
        run.path, signature=run.signature, include_failed=include_failed, strict_signature=True
    )

    # Stable sort: prioritize country / experiment_code
    def _sort_key(r: Dict[str, Any]) -> Tuple[str, str]:
        return (str(r.get('country', '')), str(r.get('experiment_code', '')))

    results = sorted(results, key=_sort_key)

    df = None
    if pd is not None:
        df = pd.DataFrame(results)

    if out_pkl:
        os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
        with open(out_pkl, 'wb') as f:
            pickle.dump(results, f)

    if out_xlsx:
        if pd is None or df is None:
            raise ModuleNotFoundError(
                "pandas is required to write .xlsx. Please run this in the notebook/conda env that has pandas installed."
            )
        os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
        df.to_excel(out_xlsx, index=False)

    return {
        'signature': run.signature,
        'checkpoint_run_dir': run.path,
        'n_results': int(len(results)),
        'n_skipped_files': int(len(skipped)),
        'out_pkl': out_pkl,
        'out_xlsx': out_xlsx,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description='Merge per-country-per-experiment checkpoint files.')
    parser.add_argument('--checkpoint_base_dir', required=True, help='e.g. results/checkpoints')
    parser.add_argument('--signature', default='latest', help='signature string or latest')
    parser.add_argument('--out_pkl', default=None, help='output pkl path')
    parser.add_argument('--out_xlsx', default=None, help='output xlsx path')
    parser.add_argument('--include_failed', action='store_true', help='include failed results too')
    args = parser.parse_args()

    summary = merge_checkpoints(
        checkpoint_base_dir=args.checkpoint_base_dir,
        signature=args.signature,
        out_pkl=args.out_pkl,
        out_xlsx=args.out_xlsx,
        include_failed=args.include_failed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    _main()
