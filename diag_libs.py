"""List which OpenMP runtime each compiled extension in this environment links against.

On macOS the question that matters is whether one process loads more than one libomp / libiomp5 copy.
Two runtimes with separate thread-pool state in one process is the working hypothesis for the CI crash.

    python diag_libs.py
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ["numpy", "scipy", "numba", "sklearn", "lightgbm", "xgboost", "catboost"]
OMP_MARKERS = ("libomp", "libiomp5", "libgomp")


def _extension_files(pkg_dir: Path) -> list[Path]:
    """Compiled extension modules and shared libraries under a package directory."""
    patterns = ("*.so", "*.dylib", "*.pyd")
    return sorted(p for pat in patterns for p in pkg_dir.rglob(pat))


def _linked_omp(path: Path) -> list[str]:
    """OpenMP runtime load commands of one binary (macOS only)."""
    out = subprocess.run(["otool", "-L", str(path)], capture_output=True, text=True, check=False).stdout
    return [line.strip().split(" ")[0] for line in out.splitlines()[1:] if any(m in line for m in OMP_MARKERS)]


def main() -> int:
    """Print every extension's OpenMP links, then the distinct runtimes."""
    print(f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    if platform.system() != "Darwin":
        print("otool is macOS-only; run this on the macOS runner.")
        return 0
    runtimes: dict[str, list[str]] = {}
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"{name}: not installed")
            continue
        pkg_dir = Path(mod.__file__).parent
        for binary in _extension_files(pkg_dir):
            for lib in _linked_omp(binary):
                runtimes.setdefault(lib, []).append(f"{name}:{binary.relative_to(pkg_dir)}")
    for lib, users in sorted(runtimes.items()):
        print(f"\n{lib}  <- {len(users)} binaries")
        for u in users[:8]:
            print(f"    {u}")
    print(f"\ndistinct OpenMP runtime paths linked: {len(runtimes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
