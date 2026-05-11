"""Run main.py for multiple n_topics values in sequence.

Place this file in src/ next to main.py and execute from the project root:
    python src/run.py
    python src/run.py --n_topics_list 10,30,50,100
    python src/run.py --encoder paraphrase-mpnet-base-v2 --no_classify
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run KeyNMF for multiple n_topics values")
    p.add_argument("--n_topics_list", type=str, default="10,30,50",
                   help="Comma-separated list of n_topics values")
    p.add_argument("--encoder",     type=str,   default="paraphrase-MiniLM-L3-v2")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--top_n",       type=int,   default=15)
    p.add_argument("--min_df",      type=int,   default=5)
    p.add_argument("--max_df",      type=float, default=0.5)
    p.add_argument("--no_classify", action="store_true", help="Skip classifier step")
    p.add_argument("--downsample",  action="store_true", help="Downsample majority class for classifier")
    return p.parse_args()


def main():
    args = parse_args()
    n_topics_values = [int(n.strip()) for n in args.n_topics_list.split(",")]
    main_path = Path(__file__).parent.resolve() / "main.py"

    print("=" * 50)
    print(f"n_topics values : {n_topics_values}")
    print(f"encoder         : {args.encoder}")
    print(f"seed            : {args.seed}")
    print(f"classify        : {'off' if args.no_classify else 'on'}")
    print(f"downsample      : {args.downsample}")
    print("=" * 50)

    for n in n_topics_values:
        print(f"\n>>> Running n_topics={n}")
        cmd = [
            sys.executable, str(main_path),
            "--n_topics", str(n),
            "--encoder",  args.encoder,
            "--seed",     str(args.seed),
            "--top_n",    str(args.top_n),
            "--min_df",   str(args.min_df),
            "--max_df",   str(args.max_df),
        ]
        if not args.no_classify:
            cmd.append("--classify")
        if args.downsample:
            cmd.append("--downsample")

        try:
            result = subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(130)

        if result.returncode != 0:
            print(f"\nRun for n_topics={n} failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print("\nAll runs complete. Logs and plots are in out/")


if __name__ == "__main__":
    main()
