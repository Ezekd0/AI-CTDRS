# AI-Driven Cyber Threat Detection and Response System (AI-CTDRS)

Production-ready application source and deployment configuration. This archive does not contain raw datasets, trained model binaries, or production credentials. Those must be provisioned separately before a production deployment can become prediction-ready.

## Layout
- `backend/` FastAPI app (`app/core`, `app/db`, `app/api/v1`, `app/modules/*`) and tests
- `ml/` per-dataset pipelines (CICIDS2017, CSE-CIC-IDS2018, NSL-KDD are never concatenated), training, evaluation, explainability, artifacts
- `frontend/` React + Tailwind (stage 2+)
- `reports/`, `tests/`, `docs/` (including deployment and API documentation)

## Prerequisites
Python 3.11+, Node.js 18+, Docker (for PostgreSQL).

## Backend setup (virtual environment)
```bash
cp .env.example .env            # set SECRET_KEY and DB password
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
docker compose up -d db
cd backend
uvicorn app.main:app --reload --port 8000
```
Local health check: http://localhost:8000/health. Production health must use the deployed HTTPS URL. API documentation is disabled by default in production and can be enabled with `ENABLE_API_DOCS=true`.

## Frontend setup
```bash
cd frontend
cp .env.example .env
npm install
npm run dev                     # http://localhost:5173
```

## Tests
```bash
PYTHONPATH=. pytest
```

Response actions default to simulation mode (`RESPONSE_MODE=SIMULATION`).
No datasets are included; place raw files under `ml/data/raw/<dataset>/` yourself.

## CICIDS2017 preprocessing
Put the CICIDS2017 CSV files in `ml/data/raw/CIC-IDS2017/`, then from the project root:
```bash
python -m ml.preprocessing.cicids2017_preprocessor --raw-dir ml/data/raw/CIC-IDS2017
PYTHONPATH=. pytest
```
Outputs: `ml/data/processed/cicids2017/{train,test}.csv.gz`, `ml/artifacts/cicids2017/preprocessor.joblib`
and `preprocessing_report.json` (all statistics are computed from your data at run time).

## CSE-CIC-IDS2018 preprocessing
Place the CSVs (sub-folders allowed) in `ml/data/raw/CSE-CIC-IDS2018/`:
```bash
python -m ml.preprocessing.cse_cicids2018_preprocessor --raw-dir ml/data/raw/CSE-CIC-IDS2018 \
    --nrows-per-file 50000      # optional: quick dev run; the full dataset is very large
```
Outputs go to `ml/data/processed/cse_cicids2018/` and `ml/artifacts/cse_cicids2018/`.
Differences from CICIDS2017: `docs/cicids2017_vs_cse_cicids2018_preprocessing.md`.

## NSL-KDD preprocessing
Place `KDDTrain+.txt` and `KDDTest+.txt` in `ml/data/raw/NSL-KDD/`:
```bash
python -m ml.preprocessing.nsl_kdd_preprocessor --raw-dir ml/data/raw/NSL-KDD
```
Outputs: `ml/data/processed/nsl_kdd/{train,test}.csv.gz`, `ml/artifacts/nsl_kdd/`. Details: `docs/nsl_kdd_preprocessing.md`.

## Model training (Random Forest, XGBoost, LSTM)
Run the dataset's preprocessing step first (above), then from the project root:
```bash
pip install -r ml/requirements.txt
python -m ml.training.train --dataset cicids2017 --model xgboost
python -m ml.training.train --dataset nsl-kdd --model random_forest
python -m ml.training.train --dataset cicids2018 --model lstm --task binary
python -m ml.training.train --dataset all --model all          # every dataset x model x task
```
`--task` is `binary`, `multiclass` or `both` (default). Useful options: `--seed`, `--val-size`, `--max-train-rows`
(quick dev runs), `--class-weight balanced|sqrt|none`, `--param KEY=VALUE` (repeatable), `--preprocess-if-missing`.

Outputs:
- `ml/artifacts/<dataset>/<task>/<model>/` model, preprocessor, `feature_names.json`, `label_encoder.joblib`, `metadata.json`, `metrics.json` (git-ignored)
- `reports/ml/<dataset>/<task>/<model>/` metrics, classification report, confusion matrix (CSV + PNG), ROC curve (binary), `run_info.json`
- `reports/ml/summary.csv` one row per trained model, rebuilt from the metrics files

Trained models are server-side only. Tree-model bundles also retain a deterministic training-background sample for LIME so local explanations do not use the explained row as its own background. Frontend-facing code must use `ml.training.artifacts.public_summary` /
`list_public_summaries` (metrics and descriptive metadata, no file paths); never serve files from `ml/artifacts/`.
Details: `docs/ml_training.md`.

## Deployment

See `docs/deployment.md` for the supported production boundary, environment requirements, migrations, backend startup, frontend build, artifact handling, and security controls.

## PostgreSQL / SQLAlchemy persistence

The production backend uses SQLAlchemy 2.x and PostgreSQL through `DATABASE_URL`. `SECRET_KEY` is also required from the environment and is never stored in source code. Copy `.env.example` to `.env` and provide deployment-specific values; do not commit `.env`.

### Database models

The database contains `User`, `Dataset`, `Model`, `Detection`, `Alert`, `Explanation`, `ResponseAction`, and `Report` entities with UUID-style string primary keys, foreign keys, timestamps, useful indexes, and SQLAlchemy relationships. Detection records are created from real prediction requests; the development seed script intentionally creates only dataset/model configuration and no users or cybersecurity incidents.

### Migrations

From `backend/`:

```bash
alembic upgrade head
```

To generate a migration after schema changes:

```bash
alembic revision --autogenerate -m "describe change"
```

### Development seed

```bash
PYTHONPATH=. python scripts/seed_dev.py
```

The seed is configuration-only and does not create fake detections, alerts, explanations, or response actions.


## End-to-end detection workflow

The integrated production path is:

`network features -> request validation -> dataset/model artifact selection -> saved preprocessing -> model prediction -> confidence -> Detection persistence -> thresholded Alert -> SHAP/LIME Explanation persistence -> dashboard/live monitor -> Detection Details -> simulated response -> audit trail -> detection report`

Prediction requests select a public dataset key (`cicids2017`, `cicids2018`, `nsl-kdd`); the model loader resolves the training registry slug automatically, including the `cicids2018` -> `cse_cicids2018` artifact mapping. Alert creation uses `ALERT_PROBABILITY_THRESHOLD` (default `0.75`) and does not create alerts for `normal`/`benign` predictions.

The dashboard provides the prediction submission form, live detection/alert counts, and navigation into Detection Details. Detection Details loads persisted SHAP/LIME results, controlled response actions, audit history, and can generate a persisted JSON detection report. Reports are built from the persisted database evidence rather than hard-coded values.

## Integration tests

`backend/tests/test_end_to_end_workflow.py` builds a real temporary Random Forest artifact and exercises the shared prediction, persistence, SHAP, alert, response, audit, dashboard, and report components. The LIME dependency check is intentionally skipped when `lime` is absent; production installation must satisfy `backend/requirements.txt` for real LIME execution.

## Controlled incident response

The incident-response module is deliberately simulation-only. Available actions are create alert, mark investigation, simulate IP block, simulate host isolation, escalate, and close. All actions require confirmation and a reason and are persisted for auditability. No arbitrary third-party network or host-control functionality is included.

## Comprehensive test status

The project now includes backend security/API regression tests and frontend authentication tests. The executable Python suites currently report **60 passed, 1 skipped** in the supplied environment. The single skipped test is the real LIME dependency check because the `lime` package is not installed and this environment cannot access the package registry. It is intentionally not counted as passing.

Security hardening now rejects JWT algorithms other than HS256 and rejects wildcard CORS origins. JWT/password handling uses the declared `argon2-cffi` dependency and an internal HS256 implementation so backend tests do not depend on unavailable jose/passlib packages.

The frontend has a real login/logout flow, automatic token attachment, and 401 session invalidation. Frontend Vitest/Testing Library coverage is included, but it must be executed after installing the declared npm dependencies; this environment has no npm package cache/network access.

## Production deployment

The repository includes `Dockerfile.backend`, `frontend/Dockerfile`, `frontend/nginx.conf`, and `docker-compose.production.yml`. The compose file is production-like but is not a public hosting provider deployment.

### Required production variables

Copy `.env.example` to a secret-managed environment and set at minimum:

- `DATABASE_URL`
- `SECRET_KEY`
- `CORS_ORIGINS`
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` when using the bundled PostgreSQL service
- `VITE_API_BASE_URL` at frontend build time
- `ARTIFACTS_HOST_PATH` pointing at the separately provisioned trained model bundles

Generate secrets outside source control. Do not use the example values.

### Production-like commands

```bash
cp .env.example .env
# edit .env with real secret-managed values

# Verify model bundles before starting the application
PYTHONPATH=. python scripts/verify_production_artifacts.py --root ml/artifacts --require-all

# Build and start PostgreSQL, backend and frontend
docker compose -f docker-compose.production.yml build
docker compose -f docker-compose.production.yml up -d db
docker compose -f docker-compose.production.yml up -d backend frontend

# Verify
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/health/ready
```

The backend image runs `alembic upgrade head` before Uvicorn. For an externally managed PostgreSQL database, set `DATABASE_URL` to the managed connection string and run `cd backend && alembic upgrade head` during the release step instead.

### Administrator setup

After migrations, create the first administrator interactively; credentials are never seeded:

```bash
docker compose -f docker-compose.production.yml exec backend python scripts/create_admin.py --email admin@example.com
```

### Hosting

No public hosting provider, DNS name, TLS endpoint, or production database credentials are included in this repository. A public URL must only be documented after a real provider deployment and external health check.
