
from pathlib import Path
import pandas as pd
from joblib import load
from matcher_lib import deterministic_filter, add_availability_features, greedy_assign

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--supervisors_csv", default="Supervision Matching Program - Supervisors.csv")
ap.add_argument("--associates_csv",  default="Supervision Matching Program - Associates.csv")
ap.add_argument("--model_path",      default="supervision_pair_model.joblib")
ap.add_argument("--out_matches",     default="supervision_matches.csv")
ap.add_argument("--out_unassigned",  default="associates_unassigned.csv")
args = ap.parse_args()

sup = pd.read_csv(args.supervisors_csv)
assoc = pd.read_csv(args.associates_csv)

cand = deterministic_filter(sup, assoc)
cand, model = add_availability_features(cand, assoc, sup)

score_col = 'AvailabilityScore'
try:
    bundle = load(args.model_path)
    clf = bundle['model']; feat_cols = bundle['features']
    cand['match_score'] = clf.predict_proba(cand[feat_cols])[:,1]
    cand['final_score'] = 0.5*cand['match_score'] + 0.5*cand['AvailabilityScore']
    score_col = 'final_score'
except Exception:
    pass

assigned = greedy_assign(cand, sup, score_col=score_col)
assigned.to_csv(args.out_matches, index=False)

assigned_set = set(assigned['assoc_idx'].tolist())
assoc.loc[~assoc.index.isin(assigned_set)].to_csv(args.out_unassigned, index=False)

print('Wrote ->', Path(args.out_matches).resolve())
print('Wrote ->', Path(args.out_unassigned).resolve())
