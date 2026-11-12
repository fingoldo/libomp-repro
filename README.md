# libomp-repro

Standalone reproducer for the macOS arm64 crash seen in mlframe CI: `Fatal Python error: Segmentation fault` inside
libomp's `__kmp_suspend_initialize_thread`, reached during LightGBM Dataset construction, and `OMP: Error #179`.

The crash was not gated by thread count (`OMP_NUM_THREADS=1` confirmed active), so the working hypothesis is that one
process loads more than one OpenMP runtime (`DYLD_LIBRARY_PATH` points numba at Homebrew's libomp while other wheels
bundle their own copy). `diag_libs.py` checks that directly; `repro.py` runs the stages one per process.

## Run

    python diag_libs.py              # macOS only: lists OpenMP runtimes per extension
    python repro.py numba
    python repro.py sklearn
    python repro.py lightgbm
    python repro.py combined

A stage that exits 139 or prints `Fatal Python error` has reproduced the crash.

## Status

Not yet run on macOS. Written on Windows, where none of the crash-relevant libraries use libomp in the same way.
The CI workflow in `.github/workflows/repro.yml` is manual (`workflow_dispatch`) and has not been triggered.
