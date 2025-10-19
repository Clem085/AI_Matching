
#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent

# Ensure local imports still work when invoked from another directory
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import generate_data  # noqa: E402
import build_pairs  # noqa: E402
import train_model  # noqa: E402

def _resolve(path_str: str | None, default_filename: str) -> Path:
    if path_str:
        return Path(path_str).expanduser().resolve()
    return BASE_DIR / default_filename


def do_generate(args):
    print("=== generate ===")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else BASE_DIR
    generate_data.main(out_dir=out_dir)
    print("done.\n")

def do_build(args):
    print("=== build ===")
    sup = _resolve(args.supervisors_csv, "Supervision_Supervisors_SYNTH.csv")
    asc = _resolve(args.associates_csv, "Supervision_Associates_SYNTH.csv")
    out = _resolve(args.pairs_csv, "Supervision_HistoricalPairs_SYNTH.csv")
    build_pairs.main(sup_csv=sup, assoc_csv=asc, out_csv=out)
    print("done.\n")

def do_train(args):
    print("=== train ===")
    pairs = _resolve(args.pairs_csv, "Supervision_HistoricalPairs_SYNTH.csv")
    model_out = _resolve(args.model_out, "supervision_pair_model.joblib")
    train_model.main(pairs_csv=pairs, model_out=model_out)
    print("done.\n")

def do_score(args):
    print("=== score ===")
    score_script = BASE_DIR / "score_matches.py"
    cmd = [
        sys.executable, str(score_script),
        "--supervisors_csv", str(_resolve(args.supervisors_csv, "Supervision_Supervisors_SYNTH.csv")),
        "--associates_csv", str(_resolve(args.associates_csv, "Supervision_Associates_SYNTH.csv")),
        "--model_path", str(_resolve(args.model_out, "supervision_pair_model.joblib")),
        "--out_matches", str(_resolve(args.out_matches, "supervision_matches.csv")),
        "--out_unassigned", str(_resolve(args.out_unassigned, "associates_unassigned.csv")),
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
    parser.add_argument("--out_dir", default=str(BASE_DIR), help="[generate] output directory")
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
