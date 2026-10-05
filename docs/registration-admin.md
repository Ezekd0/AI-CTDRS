# Registration and administration

The sign-up form at `/signup` (also `/register`) sends `full_name`, `email`, and `password` to the existing `POST /api/auth/register`. The existing SQLAlchemy session commits the `User` to the configured `DATABASE_URL`. Email is normalized to lowercase, passwords use the existing Argon2 hasher, and self-registration always assigns the `viewer` role. Duplicate emails return HTTP 409, including concurrent requests. Password confirmation is checked in the frontend and is not stored or sent to the API.

Migration `0005_user_full_name` adds a nullable `VARCHAR(200)` to the existing `users` table. The frontend requires a nonblank name. The API accepts omitted names for compatibility with existing clients; existing accounts retain a null name and their credentials, roles, timestamps, and status. Provided names must be nonblank and at most 200 characters. Passwords retain the existing 12–128 character requirement.

## Deployment setup

Run the existing migration release step, `cd backend && alembic upgrade head`, against the intended database before serving the updated backend. The existing Docker startup already runs this command. Dockerfile, Render configuration, PostgreSQL settings, production dependencies, and PYTHONPATH are unchanged.

Set `VITE_API_URL` at frontend build time to the backend API base including `/api`. If the frontend and API share an origin, the default `/api` is sufficient. Existing `VITE_API_BASE_URL` deployments remain supported as a fallback. Configure the existing backend `CORS_ORIGINS` for the frontend origin. The existing Vercel SPA rewrite supports direct visits to the new routes.

Use an existing administrator account, or the existing interactive bootstrap script from `backend`:

```sh
PYTHONPATH=..:. python scripts/create_admin.py --email admin@example.com
```

No administrator credentials are seeded. Users can sign in at `/login` and access `/dashboard`. Ordinary authenticated accounts are redirected from `/admin` and `/admin/users` to `/dashboard`; unauthenticated visits are redirected to `/login`. Sessions are verified with the existing `/api/auth/me`, rather than trusting a stored role.

## Admin API

All new endpoints use `require_roles('administrator')` on the router. Authentication reads the current user from the database on every request, so status and role changes apply to existing tokens. Missing/invalid authentication returns 401; authenticated nonadministrators receive 403.

- `GET /api/admin/overview`: user counts, active accounts, last-seven-day registrations, recent registrations/detections, and API/database/response-mode status. Active means the account is enabled (`is_active`), not currently online.
- `GET /api/admin/users`: paginated users with `search`, `role`, `is_active`, `offset`, and `limit` (maximum 100). Search matches name/email literally and ignores case. Password hashes are never returned.
- `PATCH /api/admin/users/{user_id}/status`: accepts `{"is_active": true}` or `false`. Disabled accounts cannot log in or use existing tokens. Administrator accounts cannot be disabled here (409) to avoid administrator lockout.

The existing `PATCH /api/auth/users/{user_id}/role` is unchanged. No privilege-management endpoint is exposed by sign-up.

## Validation

```sh
cd frontend
npm install
npm test
npm run build
cd ..
PYTHONPATH=backend:. python -m pytest backend/tests -q
git diff --check
```

To exercise persistence and concurrent registration against PostgreSQL, use a **dedicated disposable database**, apply all migrations, and run:

```sh
CTDRS_TEST_DATABASE_URL='postgresql://USER@HOST/TEST_DB' \
  PYTHONPATH=backend:. python -m pytest backend/tests/test_registration_admin.py -q
```

The PostgreSQL fixture expects a migrated schema and removes its `registration-*` test accounts afterward. Never point it at a production database.
