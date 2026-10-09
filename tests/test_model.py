from pathlib import Path
import joblib
import pandas as pd

def test_saved_model_predicts():
    path = Path(__file__).resolve().parents[1] / "models" / "failure_pipeline.joblib"
    artifact = joblib.load(path)
    row = pd.DataFrame([[300.0, 310.0, 1500.0, 40.0, 100.0]], columns=artifact["features"])
    pred = artifact["pipeline"].predict(row)
    prob = artifact["pipeline"].predict_proba(row)
    assert len(pred) == 1
    assert pred[0] in (0, 1)
    assert prob.shape == (1, 2)
