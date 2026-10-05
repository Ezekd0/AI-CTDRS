# Production FastAPI backend

## API surface

The production API is rooted at `/api` and is split into routers for `/auth`, `/dashboard`, `/datasets`, `/models`, `/predict`, `/detections`, `/alerts`, `/explain`, `/reports`, and `/responses`. OpenAPI is available at `/docs`; ReDoc is available at `/redoc`; health is `/api/health`.

## Prediction flow

`POST /api/predict` accepts a dataset/task/model selector and raw feature values. The server validates the request, loads the selected trained bundle from the configured artifact root, loads the saved `preprocessor.joblib`, transforms the features, executes the selected model, persists the detection, and optionally invokes SHAP and/or LIME. The request never supplies a filesystem path.

The artifact loader and prediction service validate path components and keep all model/preprocessor files server-side. Arbitrary filesystem access is not exposed through API parameters.

## Configuration

Configuration comes from environment variables via Pydantic Settings. Important values include `DATABASE_URL`, `SECRET_KEY`, `CORS_ORIGINS`, `ARTIFACTS_ROOT`, `REPORTS_ROOT`, `LOG_LEVEL`, `ENABLE_EXPLANATIONS`, and `MAX_PREDICTION_FEATURES`.

## Error handling and logging

Validation failures return HTTP 422, application value errors return HTTP 422, missing resources return HTTP 404 from their routers, and unexpected exceptions are logged server-side and returned as a generic HTTP 500 response without filesystem details.

## Authentication and authorization

Authentication uses Argon2 password hashing and short-lived JWT bearer access tokens. New self-registered accounts receive the `viewer` role. An existing administrator can change roles through `PATCH /api/auth/users/{user_id}/role`.

Roles:
- `administrator`: model training, dataset administration, response actions, and privileged user-role management.
- `analyst`: predictions, detections, explanations, reporting, and authorized response actions.
- `viewer`: read-only dashboards, model results, detections, and explanations.

`POST /api/auth/logout` records the token `jti` in `revoked_tokens` until token expiry. Every authenticated request checks this revocation list.

Create the first administrator interactively after migrations:

```bash
python -m scripts.create_admin --email admin@example.com
```

No administrator password is seeded in source control or development seed data.

## Dashboard aggregation

`GET /api/dashboard?start=<ISO-8601>&end=<ISO-8601>` returns production dashboard aggregates from PostgreSQL. `total_events` is defined as persisted `Detection` records because the current schema has no separate raw-event table. The endpoint returns severity counts, daily detection trend, attack/prediction distribution, recent detections, recent alerts, and the currently active model. Empty result sets are returned as empty arrays/zero counts; the API never inserts demo incidents to populate the dashboard.


### Controlled incident response

Incident response is **SIMULATION-only by default and in the current implementation** (`RESPONSE_MODE=SIMULATION`). Administrator and analyst roles can execute the six controlled actions only after explicit confirmation and a reason are supplied. Every action is persisted in `response_actions` with the authenticated user, incident/detection ID, action, timestamp, status, and reason. IP-block and host-isolation actions only create an auditable simulation record and never call an external system. A dedicated `app/core/response_integrations.py` boundary is reserved for a future, separately authorized defensive laboratory integration; it is not enabled by this application.

Endpoints:
- `GET /api/responses?detection_id=<id>` — response history.
- `POST /api/responses/{detection_id}/actions` — execute a confirmed response action.

The request must contain `confirmed: true` and a non-empty reason. Viewer users cannot execute response actions.
