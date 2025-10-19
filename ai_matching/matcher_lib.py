
from __future__ import annotations
from typing import Set
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalize_state(s: str) -> str:
    return str(s).strip().upper()

def parse_state_list(s: str) -> Set[str]:
    if pd.isna(s):
        return set()
    parts = [p.strip().upper() for p in str(s).replace(';', ',').split(',') if p.strip()]
    return set(parts)

def parse_license_list(s: str) -> Set[str]:
    if pd.isna(s):
        return set()
    s = str(s)
    parts = [p.strip().upper() for p in s.replace(';', ',').split(',') if p.strip()]
    return set(parts) if parts else {s.strip().upper()}

def coerce_int(x, default=1) -> int:
    try:
        v = int(x)
        return v if v >= 0 else default
    except Exception:
        return default

def availability_overlap(a: str, b: str) -> int:
    aset = set([p.strip().upper() for p in str(a).split(',') if p.strip()])
    bset = set([p.strip().upper() for p in str(b).split(',') if p.strip()])
    return len(aset & bset)

class AvailabilityModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2))
        self._fit = False
    def fit(self, assoc_avail: pd.Series, sup_avail: pd.Series):
        all_text = pd.concat([assoc_avail.fillna('').astype(str),
                              sup_avail.fillna('').astype(str)]).values
        self.vectorizer.fit(all_text)
        self._fit = True
    def score(self, a: str, b: str) -> float:
        if not self._fit:
            raise RuntimeError('AvailabilityModel must be fit() before scoring.')
        A = self.vectorizer.transform([str(a)])
        B = self.vectorizer.transform([str(b)])
        return float(cosine_similarity(A, B).ravel()[0])

def deterministic_filter(supervisors: pd.DataFrame, associates: pd.DataFrame) -> pd.DataFrame:
    sup = supervisors.copy()
    assoc = associates.copy()
    sup['State_set'] = sup['State'].map(parse_state_list)
    assoc['State_set'] = assoc['State'].map(parse_state_list)
    sup['State_norm'] = sup['State_set'].map(lambda st: ','.join(sorted(st)))
    assoc['State_norm'] = assoc['State_set'].map(lambda st: ','.join(sorted(st)))
    sup['WhoSet'] = sup['Who can you supervise?'].map(parse_license_list)
    assoc['License_norm'] = assoc['License Type'].astype(str).str.strip().str.upper()
    rows = []
    for ai, a in assoc.iterrows():
        for sj, s in sup.iterrows():
            if not a['State_set'] or not s['State_set']:
                continue
            if not (a['State_set'] & s['State_set']):
                continue
            if a['License_norm'] not in s['WhoSet']:
                continue
            rows.append({
                'assoc_idx': ai,
                'sup_idx': sj,
                'Associate': a['Name'],
                'Associate Email': a['Email Address'],
                'Associate State': a['State'],
                'Associate License': a['License Type'],
                'Associate Availability': a.get('Availability',''),
                'Supervisor': s['Name'],
                'Supervisor Email': s['Email Address'],
                'Supervisor State': s['State'],
                'Who can you supervise?': s['Who can you supervise?'],
                'Supervisor Availability': s.get('Availability',''),
                'Capacity': s.get('Capacity', 1),
            })
    import pandas as pd
    return pd.DataFrame(rows)

def add_availability_features(candidates: pd.DataFrame, assoc_df: pd.DataFrame, sup_df: pd.DataFrame, model: AvailabilityModel | None = None):
    if model is None:
        model = AvailabilityModel()
        model.fit(assoc_df['Availability'], sup_df['Availability'])
    scores, overlaps = [], []
    for _, r in candidates.iterrows():
        a_text = r['Associate Availability']
        s_text = r['Supervisor Availability']
        overlaps.append(availability_overlap(a_text, s_text))
        scores.append(model.score(a_text, s_text))
    cand = candidates.copy()
    cand['AvailabilityOverlap'] = overlaps
    cand['AvailabilityScore'] = scores
    return cand, model

def greedy_assign(candidates: pd.DataFrame, sup_df: pd.DataFrame, score_col: str = 'AvailabilityScore') -> pd.DataFrame:
    sup_capacity = {i: coerce_int(sup_df.loc[i, 'Capacity']) for i in sup_df.index}
    assigned_assoc = set()
    picks = []
    for _, row in candidates.sort_values(score_col, ascending=False).iterrows():
        ai = int(row['assoc_idx']); sj = int(row['sup_idx'])
        if ai in assigned_assoc:
            continue
        if sup_capacity.get(sj, 0) <= 0:
            continue
        assigned_assoc.add(ai)
        sup_capacity[sj] = sup_capacity.get(sj, 0) - 1
        picks.append(row)
    import pandas as pd
    return pd.DataFrame(picks).reset_index(drop=True)
