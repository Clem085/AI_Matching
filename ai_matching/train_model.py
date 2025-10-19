
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main(pairs_csv='Supervision_HistoricalPairs_SYNTH.csv', model_out='supervision_pair_model.joblib'):
    pairs = pd.read_csv(pairs_csv)
    feat_cols = ['AvailabilityOverlap','AvailabilityScore','Capacity']
    if 'Capacity' not in pairs.columns and 'SupervisorCapacity' in pairs.columns:
        feat_cols = ['AvailabilityOverlap','AvailabilityScore','SupervisorCapacity']

    X = pairs[feat_cols]; y = pairs['Label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
    pipe.fit(X_train, y_train)
    print('=== Validation Report ===')
    print(classification_report(y_val, pipe.predict(X_val)))

    dump({'model': pipe, 'features': feat_cols}, model_out)
    print('Saved model ->', Path(model_out).resolve())

if __name__ == '__main__':
    main()
