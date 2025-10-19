
#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import subprocess

# Import our modules
import generate_data
import build_pairs
import train_model

def do_generate(args):
    print("=== generate ===")
    generate_data.main(out_dir=args.out_dir)
    print("done.\n")

def do_build(args):
    print("=== build ===")
    sup = args.supervisors_csv or "Supervision_Supervisors_SYNTH.csv"
    asc = args.associates_csv or "Supervision_Associates_SYNTH.csv"
    out = args.pairs_csv or "Supervision_HistoricalPairs_SYNTH.csv"
    build_pairs.main(sup_csv=sup, assoc_csv=asc, out_csv=out)
    print("done.\n")

def do_train(args):
    print("=== train ===")
    pairs = args.pairs_csv or "Supervision_HistoricalPairs_SYNTH.csv"
    model_out = args.model_out or "supervision_pair_model.joblib"
    train_model.main(pairs_csv=pairs, model_out=model_out)
    print("done.\n")

def do_score(args):
    print("=== score ===")
    cmd = [
        sys.executable, "score_matches.py",
        "--supervisors_csv", args.supervisors_csv or "Supervision_Supervisors_SYNTH.csv",
        "--associates_csv", args.associates_csv or "Supervision_Associates_SYNTH.csv",
        "--model_path", args.model_out or "supervision_pair_model.joblib",
        "--out_matches", args.out_matches or "supervision_matches.csv",
        "--out_unassigned", args.out_unassigned or "associates_unassigned.csv",
    ]
    subprocess.run(cmd, check=True)
    print("done.\n")

ACTIONS = {
    "generate": do_generate,
    "build": do_build,
    "train": do_train,
    "score": do_score,
}

def main():
    parser = argparse.ArgumentParser(
        description="Supervision matching CLI: chain actions like `generate build train score`"
    )
    parser.add_argument("actions", nargs="+",
                        help="Sequence of actions to run: generate, build, train, score")
    # Global/optional parameters used by relevant actions
    parser.add_argument("--out_dir", default=".", help="[generate] output directory")
    parser.add_argument("--supervisors_csv", help="[build/score] supervisors CSV path")
    parser.add_argument("--associates_csv", help="[build/score] associates CSV path")
    parser.add_argument("--pairs_csv", help="[build/train] pairs CSV path (in/out)")
    parser.add_argument("--model_out", help="[train/score] model path (out/in)")
    parser.add_argument("--out_matches", help="[score] assigned matches CSV")
    parser.add_argument("--out_unassigned", help="[score] unassigned associates CSV")

    args = parser.parse_args()

    # Validate actions
    for a in args.actions:
        if a not in ACTIONS:
            parser.error(f"Unknown action: {a}. Choose from {list(ACTIONS)}")

    # Run actions in order
    for a in args.actions:
        ACTIONS[a](args)

if __name__ == "__main__":
    main()
