# Main administrator on Render Free

Release the updated backend through the existing deployment process before using setup. Keep the existing PostgreSQL DATABASE_URL and JWT configuration. No Shell, dependencies, or new migration are required.

Set Render API environment variables: ADMIN_BOOTSTRAP_ENABLED=true, ADMIN_BOOTSTRAP_KEY to a fresh random secret of at least 32 characters different from your JWT secret, MAIN_ADMIN_EMAIL to your exact valid email, and MAIN_ADMIN_FULL_NAME=Ospa Admin. supergmail.com is invalid; no address is invented. Save and wait for the service update.

Visit https://YOUR-RENDER-API-HOST/api/auth/bootstrap-admin. Enter the key and a chosen 12–128 character password in the masked fields, confirm the password, and submit. The form sends POST to the same URL with Authorization: Bearer <key> and JSON containing only password. No credentials are saved in browser storage or URL parameters. HTTP 201 indicates success. HTTP 409 means an administrator or conflicting account already exists; no existing account is changed.

Immediately set ADMIN_BOOTSTRAP_ENABLED=false, remove ADMIN_BOOTSTRAP_KEY, save, and wait for the service update. The setup URL will return 404. The fixed main-admin identity and PostgreSQL transaction lock prevent duplicate bootstrap creation across workers; changing MAIN_ADMIN_EMAIL cannot create another administrator after success. Existing administrator role management remains unchanged.

On your existing frontend, use /login for administrator login, /signup for user signup, /admin for the dashboard, /admin/users for registered users, and /admin/network for network report inspection. Verify a normal user can run and save a test at /network and reload history, cannot access admin routes or another user's reports, and the administrator can inspect the saved report.
