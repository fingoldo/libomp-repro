"""Minimal reproducer for the macOS libomp crash seen in mlframe CI.

Each stage runs in a fresh process so one crash cannot hide another. Usage:

    python repro.py <stage>

Stages, in increasing order of what they load into one process:

    numba      one numba parallel=True kernel (workqueue/omp threading layer)
    sklearn    sklearn's OpenMP-backed histogram gradient boosting
    lightgbm   LightGBM Dataset construction plus a short train
    combined   numba kernel, then sklearn, then LightGBM, in one process

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

    print("numba threading layer:", numba.threading_layer(), flush=True)
    print("numba total:", total(np.ones(1_000_000)), flush=True)


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


def main(stage: str) -> int:
    """Run the named stage and return a process exit code."""
    stages = {
        "numba": [_numba_stage],
        "sklearn": [_sklearn_stage],
        "lightgbm": [_lightgbm_stage],
        "combined": [_numba_stage, _sklearn_stage, _lightgbm_stage],
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
