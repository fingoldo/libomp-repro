"""Minimal reproducer for the macOS libomp crash seen in mlframe CI.

Each stage runs in a fresh process so one crash cannot hide another. Usage:

    python repro.py <stage>

Stages, in increasing order of what they load into one process:

    numba      one numba parallel=True kernel (workqueue/omp threading layer)
    sklearn    sklearn's OpenMP-backed histogram gradient boosting
    lightgbm   LightGBM Dataset construction plus a short train
    combined   numba kernel, then sklearn, then LightGBM, in one process
    torch      a torch matmul and an autograd step (torch wheels bundle their own libomp)
    torch_lgb  torch first, then LightGBM: the order that puts two OpenMP runtimes in one process
    lgb_torch  LightGBM first, then torch
    everything torch, xgboost, catboost, polars, pyarrow, numba, sklearn, then LightGBM: the library set of an mlframe process

A stage that dies with a signal (exit code 139, "Fatal Python error: Segmentation fault",
"OMP: Error #179", or "Concurrent access has been detected") has reproduced the crash.
"""

from __future__ import annotations

import faulthandler
import sys
import time

faulthandler.enable()


def _numba_stage() -> None:
    """Run one prange kernel so numba starts its threading layer."""
    import numba
    import numpy as np

    @numba.njit(parallel=True)
    def total(a):  # pragma: no cover - compiled
        s = 0.0
        for i in numba.prange(a.size):
            s += a[i]
        return s

    # threading_layer() raises until a parallel kernel has run, so run the kernel first.
    print("numba total:", total(np.ones(1_000_000)), flush=True)
    print("numba threading layer:", numba.threading_layer(), flush=True)


def _sklearn_stage() -> None:
    """Fit sklearn's OpenMP-threaded histogram gradient boosting."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 10))
    y = X[:, 0] + rng.normal(scale=0.1, size=5000)
    HistGradientBoostingRegressor(max_iter=20).fit(X, y)
    print("sklearn HGB fit ok", flush=True)


def _lightgbm_stage() -> None:
    """Construct a LightGBM Dataset (the call that crashed in CI) and train a few rounds."""
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(1)
    X = rng.normal(size=(5000, 10))
    y = X[:, 0] + rng.normal(scale=0.1, size=5000)
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    ds.construct()
    lgb.train({"verbose": -1, "num_threads": 1}, ds, num_boost_round=5)
    print("lightgbm dataset + train ok", flush=True)


def _torch_stage() -> None:
    """Run a torch matmul and one autograd step, which start torch's OpenMP pool."""
    import torch

    a = torch.randn(512, 512, requires_grad=True)
    (a @ a).sum().backward()
    print("torch matmul + backward ok, threads:", torch.get_num_threads(), flush=True)


def _everything_stage() -> None:
    """Import the whole library set of an mlframe training process and exercise each OpenMP user once."""
    import catboost
    import polars
    import pyarrow
    import xgboost

    print("imported:", catboost.__version__, polars.__version__, pyarrow.__version__, xgboost.__version__, flush=True)
    _torch_stage()
    _numba_stage()
    _sklearn_stage()
    _lightgbm_stage()


def main(stage: str) -> int:
    """Run the named stage and return a process exit code."""
    stages = {
        "numba": [_numba_stage],
        "sklearn": [_sklearn_stage],
        "lightgbm": [_lightgbm_stage],
        "combined": [_numba_stage, _sklearn_stage, _lightgbm_stage],
        "torch": [_torch_stage],
        "torch_lgb": [_torch_stage, _lightgbm_stage],
        "lgb_torch": [_lightgbm_stage, _torch_stage],
        "everything": [_everything_stage],
    }
    if stage not in stages:
        print(f"unknown stage {stage!r}; choose from {sorted(stages)}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    for fn in stages[stage]:
        fn()
    print(f"stage {stage} finished in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "combined"))
