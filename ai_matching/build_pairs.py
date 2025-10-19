
import random
from pathlib import Path
import pandas as pd
from matcher_lib import deterministic_filter, add_availability_features, coerce_int

random.seed(42)

def main(sup_csv='Supervision_Supervisors_SYNTH.csv',
         assoc_csv='Supervision_Associates_SYNTH.csv',
         out_csv='Supervision_HistoricalPairs_SYNTH.csv'):
    sup = pd.read_csv(sup_csv)
    assoc = pd.read_csv(assoc_csv)

    cand = deterministic_filter(sup, assoc)
    cand, model = add_availability_features(cand, assoc, sup)

    def label_row(r):
        sup_cap = coerce_int(r['Capacity'])
        if sup_cap <= 0:
            return 0, 'capacity_full'
        if r['AvailabilityOverlap'] == 0 and r['AvailabilityScore'] < 0.05:
            return 0, 'availability_mismatch'
        if r['AvailabilityOverlap'] <= 1 and r['AvailabilityScore'] < 0.25:
            return 0, 'low_overlap'
        if random.random() < 0.1:
            return 0, 'sampled_negative'
        return 1, 'good_match'

    labels, reasons = [], []
    for _, row in cand.iterrows():
        y, why = label_row(row)
        labels.append(y); reasons.append(why)

    pairs = cand.copy()
    pairs['Label'] = labels
    pairs['Reason'] = reasons
    pairs.to_csv(out_csv, index=False)
    print('Wrote pairs ->', Path(out_csv).resolve())

if __name__ == '__main__':
    main()
