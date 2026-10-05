# NSL-KDD preprocessing

`ml/preprocessing/nsl_kdd_preprocessor.py` is standalone; it shares nothing with the CICIDS pipelines.
As elsewhere, the dataset facts below come from public descriptions, not from files bundled here;
`preprocessing_report.json` records what your copy actually contains.

| Aspect | NSL-KDD pipeline |
|---|---|
| Files | `KDDTrain+` and `KDDTest+` (.txt/.csv). 20-percent and `-21` variants are ignored unless passed explicitly |
| Layout | Usually headerless: 41 features + label (+ difficulty level). A header-row variant is also accepted |
| Split | The official train/test split is kept. Nothing is re-split or merged |
| Categorical | `protocol_type`, `service`, `flag`: stripped, lower-cased, missing -> `unknown`, one-hot encoded fitted on train; unseen values at test/inference become all-zero rows |
| Numeric | Coerced to numbers; invalid text and inf -> NaN; train-median imputation |
| Scaling | `log1p` on `duration`, `src_bytes`, `dst_bytes`, then StandardScaler on continuous columns only. Binary flags and one-hot columns are left unscaled. Both steps can be turned off |
| Constant columns | Removed based on train (listed in the report) |
| Binary target | `is_attack`: 0 = `normal`, 1 = any attack |
| Multiclass targets | `attack_category` (normal / DoS / Probe / R2L / U2R, via a configurable name map) and `attack_type_id` for the specific attack name. Names unseen in train get id -1 and are listed in the report. Unmapped names raise an error |
| Difficulty level | Kept as metadata for evaluation, never a feature |
| Duplicates | Removed from train only; test is untouched. Test rows identical to a train row are counted in the report, not removed |
| Inference | `NSLKDDPreprocessor.transform_features(df)`: any header casing, extra columns ignored, missing columns imputed; `scaled=False` returns original units |
| Artifacts | `ml/artifacts/nsl_kdd/preprocessor.joblib`, `preprocessing_report.json`; data in `ml/data/processed/nsl_kdd/` |

Differences from the CICIDS pipelines: no flow identifiers, no inf-heavy rate columns to speak of,
mixed categorical/numeric features, a fixed official split instead of a seeded random one, and a
label hierarchy (attack name -> category). Source and destination IPs do not exist in NSL-KDD, so
the Detection Details page will need another source for them when using this dataset.
