from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "models" / "failure_pipeline.joblib"
st.set_page_config(page_title="Machine Failure Early-Warning System", page_icon="⚙️", layout="wide")
st.title("⚙️ Machine Failure Early-Warning System")
st.warning("Experimental educational demo using the AI4I synthetic benchmark. Do not use for industrial safety or maintenance decisions.")
if not ARTIFACT.exists():
    st.error("Model artifact missing. Run `python src/train.py` from the project folder first.")
    st.stop()
artifact = joblib.load(ARTIFACT)
pipe, features = artifact["pipeline"], artifact["features"]
st.caption(f"Selected model: {artifact['model_name']} • trained on AI4I 2020 benchmark")
with st.sidebar:
    st.header("Machine sensor readings")
    values = {
      features[0]: st.number_input("Air temperature (K)", 250.0, 350.0, 300.0, 0.1),
      features[1]: st.number_input("Process temperature (K)", 250.0, 400.0, 310.0, 0.1),
      features[2]: st.number_input("Rotational speed (rpm)", 500.0, 4000.0, 1500.0, 1.0),
      features[3]: st.number_input("Torque (Nm)", 0.0, 100.0, 40.0, 0.1),
      features[4]: st.number_input("Tool wear (min)", 0.0, 300.0, 100.0, 1.0),
    }
    submit = st.button("Estimate failure risk", type="primary")
if submit:
    if values[features[1]] < values[features[0]]:
        st.error("Input validation: process temperature should be at least air temperature in this benchmark.")
    else:
        row = pd.DataFrame([values], columns=features)
        probability = float(pipe.predict_proba(row)[0, 1])
        predicted = int(probability >= 0.5)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated failure probability", f"{probability:.1%}")
            st.progress(probability)
        with col2:
            if predicted:
                st.warning("Elevated risk: failure predicted")
            else:
                st.success("Lower risk: failure not predicted")
        st.caption("This probability is an experimental model score, not a calibrated real-world risk estimate.")
st.divider()
st.subheader("Held-out test evaluation")
metrics = artifact["all_metrics"]
cols = st.columns(5)
for col, label in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
    col.metric(label.replace("_", " ").title(), f"{metrics[artifact['model_name']][label]:.3f}")
st.caption("Failure recall is the share of actual failures detected. Accuracy can look high when failures are rare, even if many failures are missed.")
fig, ax = plt.subplots(figsize=(5, 3))
ConfusionMatrixDisplay(
    confusion_matrix=np.asarray(metrics[artifact["model_name"]]["confusion_matrix"]),
    display_labels=["No failure", "Failure"],
).plot(ax=ax, colorbar=False)
st.pyplot(fig)
if artifact["model_name"] == "Random Forest":
    importance = pipe.named_steps["model"].feature_importances_
else:
    importance = abs(pipe.named_steps["model"].coef_[0])
st.subheader("Feature importance / coefficient magnitude")
imp = pd.Series(importance, index=features).sort_values()
st.bar_chart(imp)
st.caption("Random Forest uses impurity-based importance; Logistic Regression shows absolute standardized coefficient magnitude. Neither proves causation.")
