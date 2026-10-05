# CICIDS2017 vs CSE-CIC-IDS2018 preprocessing

Two separate pipelines:
`ml/preprocessing/cicids2017_preprocessor.py` and `ml/preprocessing/cse_cicids2018_preprocessor.py`.
Neither assumes the other's schema, and the datasets are never concatenated.

> The "expected dataset behaviour" column reflects the public dataset descriptions and common
> community experience. It has NOT been measured here (no data is bundled). The 2018 pipeline
> records what it actually finds in your copy in `preprocessing_report.json` -> `schema`
> (columns per file, columns not shared by all files, identifier columns, repeated header rows).
> Check that report and adjust the config if your files differ.

| Aspect | CICIDS2017 pipeline | CSE-CIC-IDS2018 pipeline |
|---|---|---|
| Expected files | A handful of daily CSVs in one folder | Roughly ten daily CSVs, possibly in sub-folders; far more rows |
| Discovery | `*.csv` in the raw folder | Recursive `rglob("*.csv")`, sorted |
| Column naming | Full names with leading spaces (e.g. `Destination Port`, `Total Fwd Packets`) | Abbreviated names (e.g. `Dst Port`, `Tot Fwd Pkts`, `Flow Byts/s`). Names are normalised but never mapped onto 2017 names |
| Identifier columns | None expected in the ML CSVs; the config still drops Flow ID, IPs, source port, timestamp if present | Protocol/Timestamp expected; at least one file may carry extra Flow ID / Src IP / Src Port / Dst IP columns. Identifiers are dropped per chunk while loading, and per-file presence is reported |
| Schema consistency | Warns if files differ | Per-file column lists are recorded; columns missing from some files are reported, and features missing in too large a share of the training data are dropped (`max_missing_fraction`, default 0.5) |
| Repeated header rows | Not handled | Rows whose label equals the header text are removed and counted |
| Label text | `BENIGN`, with a corrupted dash in the Web Attack labels (fixed by collapsing punctuation) | `Benign` (different casing), hyphenated names, and the spelling `Infilteration`. A case-insensitive alias map (`label_aliases`) maps `Benign`->`BENIGN` and `Infilteration`->`Infiltration`. Different attack tools (e.g. HOIC vs LOIC) stay separate classes |
| Duplicate removal | Before identifier removal, on the raw rows | After identifier removal (so rows differing only by timestamp or IP count as duplicates), first within each file, then across files |
| Memory strategy | Whole files read at once, float64 | Chunked reading (`chunksize`), float32, per-file de-duplication, optional `max_rows_per_class` seeded cap and `--nrows-per-file` for dev runs |
| Missing / infinite values | inf -> NaN, then train-median imputation | Same, with the inf count tracked per file |
| Dropped features | all-missing, constant, correlated | `high_missing`, constant, correlated |
| Splitting, scaling, encoders, leakage rules | Leakage-safe source-file group split for CICIDS, everything learned on train only | Identical approach |
| Inference | `transform_features()` | `transform_features()`, same contract: raw headers accepted, extras ignored, missing columns imputed |
| Artifacts | `ml/artifacts/cicids2017/` | `ml/artifacts/cse_cicids2018/` |

## Shared limitations
- Source/destination IPs and timestamps are dropped from the features and not saved; the
  Detection Details stage must carry them separately.
- The two pipelines duplicate a few small helpers on purpose, so each stays standalone. They can
  be refactored behind a shared interface when the common abstraction is built.
- `destination_port` / `dst_port` is kept as a feature in both; add it to `identifier_columns`
  if you want to avoid dataset-specific port shortcuts.
