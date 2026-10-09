# Machine Failure Early-Warning System

A beginner-friendly, local ML portfolio project using the UCI AI4I 2020 Predictive Maintenance benchmark. It trains two classifiers, compares them on a held-out test set, saves the selected fitted pipeline, and provides a Streamlit dashboard for sensor-based predictions.

**Dataset caveat:** AI4I is a synthetic dataset designed to reflect industrial patterns; it is not field evidence. The dataset is provided under CC BY 4.0. The project includes the 522 KB CSV at `data/ai4i2020.csv`; the downloaded ZIP and duplicate extracted copy are not needed. Attribute it as: Matzka, S. (2020), *AI4I 2020 Predictive Maintenance Dataset*, UCI Machine Learning Repository, DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), [dataset page](https://archive.ics.uci.edu/dataset/601/ai4i). UCI permits sharing and adaptation with appropriate attribution.

## Project structure

- `data/ai4i2020.csv`: AI4I source CSV downloaded from UCI (CC BY 4.0; attribution above).
- `src/train.py`: inspect data, split, train/compare models, evaluate and save fitted pipeline.
- `models/failure_pipeline.joblib`: selected preprocessing + model artifact created by training.
- `outputs/metrics.json`: measured held-out test metrics created by training.
- `app.py`: Streamlit interactive prediction and evaluation dashboard.
- `tests/test_model.py`: smoke test for loading artifact and predicting.

## Setup and run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
pytest -q
```

If the included CSV is missing, download `ai4i2020.csv` from the official UCI dataset page above and place it at `data/ai4i2020.csv`. The script reports missing values and target counts before training. No fallback data is silently substituted: if the benchmark file is unavailable, training exits with a clear error.

## ML pipeline

The five sensor inputs are air temperature, process temperature, rotational speed, torque, and tool wear. A fixed 80/20 stratified train/test split uses seed 42. Median imputation is fitted inside each pipeline, so any learned preprocessing uses training rows only. Logistic Regression scales numeric inputs and uses `class_weight="balanced"`; Random Forest uses `class_weight="balanced_subsample"`. Models are compared on the held-out set with accuracy, failure precision/recall/F1, ROC-AUC and confusion matrices. The model with the best failure recall (F1 breaks ties) is saved. This selection rule reflects the cost of missed failures, but a real deployment needs costs, calibration, and validation agreed with domain experts.

### Actual held-out evaluation

These are the measured results from `outputs/metrics.json`, using the fixed seed 42 split. The full dataset has 10,000 rows: 9,661 no-failure and 339 failure cases (3.39%). The 2,000-row test set has 1,932 no-failure and 68 failure cases.

| Model | Accuracy | Failure precision | Failure recall | Failure F1 | ROC-AUC | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression (selected by recall) | 0.8210 | 0.1411 | 0.8382 | 0.2415 | 0.9080 | `[[1585, 347], [11, 57]]` |
| Random Forest | 0.9835 | 0.8182 | 0.6618 | 0.7317 | 0.9732 | `[[1922, 10], [23, 45]]` |

The selected Logistic Regression catches 57 of 68 test failures (11 missed) and produces 347 false alarms. Its higher failure recall comes with much lower precision than Random Forest. The selected model reflects the explicit recall-first rule in the training script; it is not necessarily the operationally best choice for a real factory.

### Reading metrics

Confusion matrix order is `[[true negatives, false positives], [false negatives, true positives]]`. Failure recall = TP/(TP+FN): low recall means missed failures. Precision = TP/(TP+FP): low precision means more false alarms. Accuracy can be misleading because the failure class is rare; predicting every row as no-failure may score well while detecting no failures. Threshold 0.5 is used for the demo; changing it trades precision against recall. ROC-AUC measures ranking across thresholds, not calibration.

Feature importance is a model explanation aid, not causation: Random Forest impurity importance is biased toward some feature structures; Logistic Regression magnitude is absolute standardized coefficient magnitude.

## Interview talking points

1. **Why stratify?** It preserves the rare failure proportion in train and test for a more representative evaluation.
2. **How is imbalance handled without leakage?** Class weights make mistakes on minority failures matter more; imputation/scaling are pipeline steps fit only during training.
3. **Why prioritize recall?** A false negative is an actual failure the warning system misses; in many maintenance contexts that is more costly than an inspection triggered by a false alarm. The costs must be confirmed with operators.
4. **Why compare precision too?** Excessive false alarms erode trust and drive unnecessary maintenance.
5. **What does the confusion matrix show?** Counts of correct no-failure/failure and the two error types; FN is the missed-failure cell.
6. **What are limitations?** AI4I is synthetic, the split is random, only five features are used, 0.5 is not optimized, and estimates are not calibrated or safety validated.
7. **Why these models?** Logistic Regression is an interpretable baseline; Random Forest captures nonlinear interactions with little feature scaling.

## Limitations

- AI4I is synthetic benchmark data and cannot establish performance on real machines.
- Evaluation uses one random stratified hold-out split, not external validation or time-based validation.
- The dashboard uses a fixed 0.5 probability threshold; the output is not calibrated, and there is no cost-based threshold selection.
- The model uses only five sensor measurements and does not account for machine identity, operating context, or changing conditions.
- Feature importance/coefficient magnitude is descriptive and does not establish causality.
- The model has not been validated for industrial use, maintenance decisions, or safety-critical systems.

## Safety

This is an experimental portfolio demo. Do not use its predictions for real industrial safety, maintenance, or operational decisions.
