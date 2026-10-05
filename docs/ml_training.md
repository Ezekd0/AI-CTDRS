# Model training

Entry point: `python -m ml.training.train --dataset {cicids2017|cicids2018|nsl-kdd|all} --model {random_forest|xgboost|lstm|all} [--task binary|multiclass|both]`

| Aspect | Behaviour |
|---|---|
| Input | Output of the dataset's own preprocessing step (`train.csv.gz`, `test.csv.gz`, `preprocessor.joblib`). Datasets are never merged |
| Test split | CICIDS2017/2018: the leakage-safe source-file group split made by preprocessing. NSL-KDD: official KDDTrain+/KDDTest+. Used once, for final evaluation |
| Validation split | Stratified hold-out from the training split (`--val-size`, default 0.1), stratified on the multiclass label for both tasks. Classes with < 2 rows stay in train. Used for early stopping and recorded validation metrics, never for the test numbers |
| Binary target | `is_attack` (0 = benign, 1 = attack); class names `[benign label, ATTACK]` |
| Multiclass target | CICIDS: `label_id` (the preprocessor's label encoder). NSL-KDD: `attack_category_id` (normal/DoS/Probe/R2L/U2R). Specific NSL-KDD attack names are not used because KDDTest+ contains names unseen in training |
| Imbalance | Class weights from training labels only (`--class-weight`): sample weights for Random Forest and XGBoost, weighted cross-entropy for the LSTM |
| Random Forest | scikit-learn, `n_jobs=-1`, seeded |
| XGBoost | `hist` trees, early stopping on the validation set, only best-iteration trees are kept and saved (`model.json`) |
| LSTM | PyTorch. The ordered feature vector is read as a short sequence (`features_per_step` values per step, default 8, zero-padded). Adam, gradient clipping, early stopping on validation loss, best weights restored (`model.pt`). This is a modelling choice: the records are not temporal |
| Reproducibility | `--seed` seeds splits, models and (LSTM) PyTorch; metadata stores the command, parameters, SHA-256 of the data/preprocessor files, library versions and git commit. GPU LSTM runs can still differ slightly between machines |

## Metrics (all computed from predictions at run time)
accuracy, balanced accuracy, precision, recall, F1 (binary: attack class; multiclass: macro, weighted also stored), Matthews correlation,
confusion matrix (counts and row-normalised), classification report, ROC-AUC (binary: P(attack); multiclass: one-vs-rest per class, macro and
support-weighted, classes without positives or negatives are listed as skipped), and PR-AUC for binary. Undefined values are stored as `null` with a reason.

## Files per model
`ml/artifacts/<dataset>/<task>/<model>/`: model file, `preprocessor.joblib`, `preprocessing_report.json`, `feature_names.json`,
`label_encoder.joblib`, `label_mapping.json`, `metadata.json`, `metrics.json`.
`reports/ml/<dataset>/<task>/<model>/`: `metrics.json`, `classification_report.{txt,json}`, `confusion_matrix{,_normalized}.csv`,
`confusion_matrix.png`, `roc_curve.{csv,png}` (binary), `precision_recall_curve.{csv,png}` (binary), `run_info.json`. Re-running overwrites that folder.

## Security
Model files are pickle/joblib/torch files. They stay under `ml/artifacts/` (git-ignored), are never written under `reports/`, and are not
served by the API. Use `public_summary()` for anything frontend-facing; `load_bundle()` is for server-side inference and rejects paths
outside the artifacts root. Only load artifacts you trained yourself.
