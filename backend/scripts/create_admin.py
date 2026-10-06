"""One-time administrator bootstrap using the normal user/authentication model."""
import argparse
import os
from getpass import getpass

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.db.models.models import User
from app.db.session import SessionLocal, get_engine
from app.schemas.auth import RegisterRequest


def bootstrap_admin(db: Session, *, email: str, password: str, full_name: str) -> str:
    # Use the same input validation and email normalization as registration/login.
    details = RegisterRequest(email=email, password=password, full_name=full_name)
    email = str(details.email).lower()
    user = db.scalar(select(User).where(User.email == email))
    action = 'updated' if user else 'created'
    if user is None:
        user = User(email=email)
        db.add(user)
    user.full_name = details.full_name
    user.password_hash = hash_password(details.password)
    user.role = 'administrator'
    user.is_active = True
    db.commit()
    return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--email', default=os.environ.get('ADMIN_EMAIL'))
    parser.add_argument('--full-name', default=os.environ.get('ADMIN_FULL_NAME'))
    args = parser.parse_args()
    email = args.email or input('Administrator email: ')
    full_name = args.full_name or input('Administrator full name: ')
    password = os.environ.get('ADMIN_PASSWORD')
    if password is None:
        password = getpass('Administrator password (12–128 characters): ')
    try:
        with SessionLocal(bind=get_engine()) as db:
            action = bootstrap_admin(db, email=email, password=password, full_name=full_name)
    except ValidationError as exc:
        # Do not print Pydantic input values: they may include the password.
        fields = ', '.join(sorted({str(e['loc'][0]) for e in exc.errors()}))
        raise SystemExit(f'Invalid administrator input: {fields}. Email must be valid, '
                         'password 12–128 characters, full name 1–200 characters.') from None
    except Exception:
        # Database exceptions can include connection details; keep them out of CLI output.
        raise SystemExit('Administrator bootstrap failed. Check database availability and migrations.') from None
    print(f'Administrator {action}. Log in through the normal login page with {email.strip().lower()}.')


if __name__ == '__main__':
    main()
