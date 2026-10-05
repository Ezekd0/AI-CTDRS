# Deployment

## Current deployment boundary

The repository provides a FastAPI backend, React/Vite frontend, PostgreSQL service definition, Alembic migrations, and server-side ML artifact loading. The supplied `docker-compose.production.yml` containerizes the backend and frontend for reproducible production-like deployment. The repository itself does **not** contain trained model binaries or production credentials.

## Production prerequisites

- Python 3.11+ for the backend/ML environment.
- Node.js 18+ for building the frontend.
- PostgreSQL 16 or a compatible PostgreSQL deployment.
- Trained artifacts under the configured `ARTIFACTS_ROOT`.
- `lime` installed if LIME generation is enabled.
- A strong random `SECRET_KEY` of at least 32 characters.
- Explicit `CORS_ORIGINS`; wildcard origins are rejected.
- `RESPONSE_MODE=SIMULATION`; external response integrations are not implemented.

## Backend

1. Create and populate `.env` from `.env.example`.
2. Install `requirements.txt`.
3. Run `alembic upgrade head` from `backend/`.
4. Start FastAPI with `PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000` from `backend/`.
5. Put a TLS-capable reverse proxy/load balancer in front of the API for production deployments.

API base path: `/api`. OpenAPI is available at `/docs` and ReDoc at `/redoc`.

## Frontend

Build with:

```bash
cd frontend
npm install
npm run build
```

Serve the generated `dist/` directory from a static web server and set `VITE_API_BASE_URL` to the deployed API origin plus `/api`.

## ML artifacts

Training is performed by the CLI in `ml.training.train`. Artifacts are server-side only and must not be exposed as static files. Random Forest and XGBoost bundles include their saved preprocessing artifact, model, label encoder, metadata, metrics, global SHAP output, and a deterministic LIME background sample.

## Database migrations

Do not rely on `Base.metadata.create_all()` for production schema management. Apply Alembic migrations before starting the application.

## Security controls

- HS256 is the only accepted JWT algorithm.
- Passwords use Argon2.
- JWT revocation is stored in the database.
- Role checks are enforced server-side.
- Response actions are simulation-only and require confirmation plus a reason.
- Model artifacts are loaded only from the configured artifact root.


## Artifact verification

Before a release, verify that every required server-side bundle is present:

```bash
PYTHONPATH=. python scripts/verify_production_artifacts.py --root ml/artifacts --require-all
```

This command fails rather than allowing a release with missing Random Forest, XGBoost, or LSTM artifacts. It validates metadata, metrics, feature names, label encoder, preprocessor, and the model binary.

## Public hosting boundary

The repository is provider-neutral. A public deployment requires an account with a hosting provider, a managed PostgreSQL instance, a TLS-enabled frontend/API endpoint, and a secure method for provisioning model artifacts. Those provider credentials and URLs are intentionally not stored in source control.
