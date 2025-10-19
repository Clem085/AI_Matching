
from pathlib import Path
import nbformat as nbf

def main(out_path="learn_from_outputs.ipynb"):
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("# Learn from Outputs — Supervision Matching"))
    cells.append(nbf.v4.new_markdown_cell("""
This notebook explains the pipeline outputs and how to read them.

**What's inside**
- The exact command you ran: `python supervision_tool.py generate build train score`
- Load artifacts:
  - `Supervision_HistoricalPairs_SYNTH.csv` (training pairs)
  - `supervision_matches.csv` and `associates_unassigned.csv`
  - `supervision_pair_model.joblib`
- Why the `UndefinedMetricWarning` happened → class imbalance and how to fix it
- Quick re-train diagnostics:
  - Validation report with `zero_division=0`
  - Confusion matrix
  - Precision–Recall + Average Precision
- Audit assignments (no double-booking, no over-capacity)
- Peek into saved model (feature weights)
- Next steps + optional extras
"""))

    cells.append(nbf.v4.new_markdown_cell("## 0) Command you ran"))
    cells.append(nbf.v4.new_code_cell("print('python supervision_tool.py generate build train score')"))

    cells.append(nbf.v4.new_markdown_cell("## 1) Load artifacts"))
    cells.append(nbf.v4.new_code_cell("""
from pathlib import Path
import pandas as pd

pairs_csv = Path('Supervision_HistoricalPairs_SYNTH.csv')
matches_csv = Path('supervision_matches.csv')
unassigned_csv = Path('associates_unassigned.csv')
model_path = Path('supervision_pair_model.joblib')

print('Found files:')
for p in [pairs_csv, matches_csv, unassigned_csv, model_path]:
    print(' -', p.resolve(), 'exists?' , p.exists())

pairs = pd.read_csv(pairs_csv) if pairs_csv.exists() else None
matches = pd.read_csv(matches_csv) if matches_csv.exists() else None
unassigned = pd.read_csv(unassigned_csv) if unassigned_csv.exists() else None

if pairs is not None:
    display(pairs.head())
if matches is not None:
    display(matches.head())
if unassigned is not None:
    display(unassigned.head())
"""))

    cells.append(nbf.v4.new_markdown_cell("## 2) Why the `UndefinedMetricWarning` (class imbalance)"))
    cells.append(nbf.v4.new_code_cell("""
if pairs is not None:
    print('Label distribution:')
    print(pairs['Label'].value_counts(dropna=False))
    print('\\nRates:')
    print(pairs['Label'].value_counts(normalize=True).rename('rate'))
else:
    print('Pairs not found.')
"""))

    cells.append(nbf.v4.new_markdown_cell("## 3) Quick re-train + diagnostics"))
    cells.append(nbf.v4.new_code_cell("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

if pairs is not None:
    feat_cols = ['AvailabilityOverlap','AvailabilityScore']
    if 'Capacity' in pairs.columns:
        feat_cols.append('Capacity')
    elif 'SupervisorCapacity' in pairs.columns:
        feat_cols.append('SupervisorCapacity')

    X = pairs[feat_cols]; y = pairs['Label']
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
    pipe.fit(X_tr, y_tr)

    print('=== Validation (zero_division=0) ===')
    print(classification_report(y_va, pipe.predict(X_va), zero_division=0))

    y_pred = pipe.predict(X_va)
    cm = confusion_matrix(y_va, y_pred, labels=[0,1])
    ConfusionMatrixDisplay(cm, display_labels=[0,1]).plot(values_format='d')
    plt.title('Confusion Matrix (Validation)'); plt.show()

    y_score = pipe.predict_proba(X_va)[:,1]
    prec, rec, thr = precision_recall_curve(y_va, y_score)
    ap = average_precision_score(y_va, y_score)
    plt.figure(); plt.plot(rec, prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR (AP={ap:.3f})'); plt.grid(True, alpha=0.3); plt.show()
else:
    print('Pairs not found.')
"""))

    cells.append(nbf.v4.new_markdown_cell("## 4) Audit assignments"))
    cells.append(nbf.v4.new_code_cell("""
if matches is not None:
    print('Rows:', len(matches))
    if 'assoc_idx' in matches.columns:
        print('Duplicate associate assignments:', matches['assoc_idx'].duplicated().sum())
    else:
        print('assoc_idx missing; skip duplicate check.')
    if 'sup_idx' in matches.columns and 'Capacity' in matches.columns:
        used = matches.groupby('sup_idx').size().rename('assigned')
        cap = matches[['sup_idx','Capacity']].drop_duplicates().set_index('sup_idx')['Capacity']
        cap_check = used.to_frame().join(cap, how='left')
        cap_check['over'] = cap_check['assigned'] - cap_check['Capacity']
        print('Any supervisors over capacity? ->', (cap_check['over']>0).any())
        display(cap_check.head(10))
    else:
        print('sup_idx/Capacity missing; skip capacity check.')
else:
    print('Matches not found.')
"""))

    cells.append(nbf.v4.new_markdown_cell("## 5) Peek inside saved model (feature weights)"))
    cells.append(nbf.v4.new_code_cell("""
from joblib import load
import pandas as pd

if model_path.exists():
    bundle = load(model_path)
    model = bundle['model']; feats = bundle['features']
    try:
        coef = model.named_steps['clf'].coef_.ravel()
        display(pd.DataFrame({'feature': feats, 'weight': coef}).sort_values('weight', ascending=False))
    except Exception as e:
        print('Could not read coefficients:', e)
else:
    print('Model not found.')
"""))

    cells.append(nbf.v4.new_markdown_cell("## 6) Next steps"))
    cells.append(nbf.v4.new_markdown_cell("""
- Add negatives / use `class_weight='balanced'` in `train_model.py`
- Parse availability into per-slot features (`Mon_AM`, `Tue_PM`, …)
- Swap greedy for optimal assignment (Hungarian/ILP) if needed
- Tune the blend between ML probability and availability similarity
"""))

    nb['cells'] = cells
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote notebook -> {Path(out_path).resolve()}")

if __name__ == "__main__":
    main()
