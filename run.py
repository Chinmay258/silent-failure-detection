"""
run.py
------
Single entry point. Trains all three conditions, evaluates, and plots.

Usage:
    python run.py                          # defaults: 10 epochs, 5 runs
    python run.py --epochs 15 --runs 3    # custom
    python run.py --condition label_noise  # single condition only
"""

import argparse
import subprocess
import sys


CONDITIONS = ["label_noise", "spurious", "dist_shift", "class_imbalance"]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--runs",      type=int, default=5)
    parser.add_argument("--condition", type=str, default="all",
                        choices=CONDITIONS + ["all"])
    parser.add_argument("--results",   type=str, default="results")
    parser.add_argument("--figures",   type=str, default="figures")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip training, only evaluate and plot")
    args = parser.parse_args()

    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    # --- Train ---
    if not args.skip_train:
        for cond in conditions:
            run([
                sys.executable, "train.py",
                "--condition", cond,
                "--epochs",    str(args.epochs),
                "--runs",      str(args.runs),
                "--out",       args.results,
            ])

    # --- Evaluate ---
    run([sys.executable, "evaluate.py", "--results", args.results])

    # --- Plot ---
    run([sys.executable, "plot.py",
         "--results", args.results,
         "--out",     args.figures])

    print("\n✓ All done.")
    print(f"  Results : {args.results}/")
    print(f"  Figures : {args.figures}/main_results.png")


if __name__ == "__main__":
    main()
