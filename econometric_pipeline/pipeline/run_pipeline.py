"""
run_pipeline.py
════════════════════════════════════════════════════════════════════════
Master runner — executes all six modules in sequence.

Usage
─────
    python run_pipeline.py [--data PATH_TO_CSV] [--skip 4 5]

Arguments
─────────
  --data   PATH   Path to panel CSV (default: panel_ready_for_modeling.csv
                  in the same directory as this script).
  --skip   LIST   Module numbers to skip (e.g. --skip 4 to skip bootstrap
                  if running a quick test).
  --quiet         Suppress per-module separator banners.

Output
──────
All figures, tables, and JSON summaries are written to:
  ./figures/
  ./tables/
  ./json/

Prerequisites
─────────────
    pip install linearmodels statsmodels scikit-learn shap pandas numpy \
                matplotlib scipy arch joblib
════════════════════════════════════════════════════════════════════════
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

MODULES = [
    ("01", "01_data_preparation.py",   "Data Preparation & Institution Index"),
    ("02", "02_panel_estimation.py",   "Panel Estimation (Two-Way FE)"),
    ("03", "03_diagnostics.py",        "Panel Diagnostics"),
    ("04", "04_bootstrap_inference.py","Wild Cluster Bootstrap Inference"),
    ("05", "05_robustness.py",         "Robustness Checks"),
    ("06", "06_ml_triangulation.py",   "ML Triangulation"),
]

def run_module(script: Path, skip: list, quiet: bool) -> bool:
    num = script.name[:2]
    if num in skip:
        print(f"\n  [SKIPPED] Module {num}: {script.name}")
        return True

    if not quiet:
        print(f"\n{'═' * 70}")
        print(f"  RUNNING MODULE {num}: {script.name}")
        print(f"{'═' * 70}")

    t0  = time.time()
    ret = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(script.parent),
    )
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"\n  ✗ Module {num} FAILED (return code {ret.returncode})")
        return False

    print(f"\n  ✓ Module {num} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full econometric pipeline.")
    parser.add_argument("--data",  type=str,  default=None,
                        help="Path to panel CSV.")
    parser.add_argument("--skip",  type=str,  nargs="*", default=[],
                        help="Module numbers to skip (e.g. --skip 4 5).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress banner output.")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Validate data path
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"✗ Data file not found: {data_path}")
            sys.exit(1)
        # Copy or symlink to expected location
        import shutil
        dest = base_dir / "panel_ready_for_modeling.csv"
        if data_path.resolve() != dest.resolve():
            shutil.copy(data_path, dest)
            print(f"✓ Data copied → {dest}")
    else:
        data_path = base_dir / "panel_ready_for_modeling.csv"
        if not data_path.exists():
            print(f"✗ No data file found at {data_path}")
            print("  Use --data PATH to specify the CSV location.")
            sys.exit(1)

    print("\n" + "═" * 70)
    print("  ECONOMETRIC PIPELINE: Violence → Institutions → FDI → Growth")
    print("  Central America, Colombia, Dominican Republic | 2000–2024")
    print("═" * 70)
    print(f"  Data:    {data_path}")
    print(f"  Skipping modules: {args.skip if args.skip else 'none'}")

    t_start = time.time()
    results = []

    for num, filename, description in MODULES:
        script = base_dir / filename
        if not script.exists():
            print(f"\n  ✗ Script not found: {script}")
            results.append((num, description, False, 0))
            continue

        success = run_module(script, args.skip, args.quiet)
        results.append((num, description, success, 0))

        if not success:
            print(f"\n  Pipeline halted at Module {num}.")
            print("  Fix the error and re-run, or use --skip {num} to bypass.")
            break

    total = time.time() - t_start
    print(f"\n{'═' * 70}")
    print("  PIPELINE SUMMARY")
    print(f"{'═' * 70}")
    for num, desc, ok, _ in results:
        status = "✓" if ok else "✗" if num not in args.skip else "—"
        print(f"  {status}  Module {num}: {desc}")
    print(f"\n  Total runtime: {total:.1f}s")
    print(f"\n  Output directories:")
    print(f"    Figures → {base_dir / 'figures'}/")
    print(f"    Tables  → {base_dir / 'tables'}/")
    print(f"    JSON    → {base_dir / 'json'}/")
    print("═" * 70)


if __name__ == "__main__":
    main()
