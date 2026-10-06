"""One-time main administrator setup using the existing authentication system."""
import hmac
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.security import hash_password
from app.db.models.models import User
from app.db.session import get_db
from app.schemas.auth import UserResponse

MAIN_ADMIN_ID = '00000000-0000-4000-8000-000000000001'
router = APIRouter(prefix='/auth/bootstrap-admin', tags=['Authentication'], include_in_schema=False)


def enabled_settings(settings=Depends(get_settings)):
    if not settings.admin_bootstrap_enabled:
        raise HTTPException(404, 'Not found')
    return settings


class BootstrapRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    password: SecretStr = Field(min_length=12, max_length=128)


@router.post('', response_model=UserResponse, status_code=201)
def create_main_admin(body: BootstrapRequest, authorization: str | None = Header(default=None),
                      settings=Depends(enabled_settings), db: Session = Depends(get_db)):
    supplied = (authorization or '').removeprefix('Bearer ')
    if not settings.admin_bootstrap_key or not hmac.compare_digest(
        supplied.encode(), settings.admin_bootstrap_key.get_secret_value().encode()
    ):
        raise HTTPException(403, 'Invalid bootstrap key')
    # Serialize PostgreSQL workers without new infrastructure or database tables.
    if db.get_bind().dialect.name == 'postgresql':
        db.execute(text('SELECT pg_advisory_xact_lock(1129595986)'))
    email = str(settings.main_admin_email).lower()
    if db.scalar(select(User.id).where(or_(User.role == 'administrator', User.id == MAIN_ADMIN_ID))):
        raise HTTPException(409, 'Main administrator setup is already completed')
    if db.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(409, 'Configured email already belongs to an account; no account was changed')
    user = User(id=MAIN_ADMIN_ID, full_name=settings.main_admin_full_name.strip(), email=email,
                password_hash=hash_password(body.password.get_secret_value()), role='administrator', is_active=True)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, 'Administrator setup conflicts with an existing account') from None
    db.refresh(user)
    return user


@router.get('', response_class=HTMLResponse)
def setup_form(settings=Depends(enabled_settings)):
    return HTMLResponse('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Main administrator setup</title><h1>Create the main administrator</h1>
<p>The email and full name come from the Render environment configuration.</p>
<form id="setup">
<p><label>Bootstrap key <input id="key" type="password" autocomplete="off" required></label></p>
<p><label>Admin password <input id="password" type="password" autocomplete="new-password" minlength="12" maxlength="128" required></label></p>
<p><label>Confirm password <input id="confirm" type="password" autocomplete="new-password" required></label></p>
<button>Create main administrator</button></form><p id="result" role="status"></p>
<script>
const form = document.getElementById('setup');
form.addEventListener('submit', async event => {
  event.preventDefault();
  const key = document.getElementById('key');
  const password = document.getElementById('password');
  const confirm = document.getElementById('confirm');
  const result = document.getElementById('result');
  if (password.value !== confirm.value) { result.textContent = 'Passwords must match.'; return; }
  const button = form.querySelector('button'); button.disabled = true;
  try {
    const response = await fetch(location.pathname, {method:'POST', credentials:'omit',
      headers:{'Content-Type':'application/json', 'Authorization':'Bearer ' + key.value},
      body:JSON.stringify({password:password.value})});
    result.textContent = response.status === 201
      ? 'Administrator created. Immediately disable ADMIN_BOOTSTRAP_ENABLED in Render, then sign in through the normal login page.'
      : 'Setup failed (HTTP ' + response.status + '). Check configuration; existing accounts are never changed.';
    if (response.status === 201) form.hidden = true;
  } catch { result.textContent = 'Request failed. Check your connection before retrying.'; }
  finally { key.value = ''; password.value = ''; confirm.value = ''; button.disabled = false; }
});
</script></html>''', headers={'Cache-Control': 'no-store', 'Content-Security-Policy':
    "default-src 'none'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'none'; frame-ancestors 'none'; base-uri 'none'"})
