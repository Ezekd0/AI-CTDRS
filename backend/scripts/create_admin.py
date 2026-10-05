"""Create an administrator without embedding credentials in source or seed data."""
import argparse
from getpass import getpass
from sqlalchemy import select
from app.core.security import hash_password
from app.db.session import get_db
from app.db.models.models import User

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--email', required=True); args=parser.parse_args()
    password=getpass('Administrator password: ')
    if len(password) < 12: raise SystemExit('Password must be at least 12 characters')
    db=next(get_db())
    try:
        if db.scalar(select(User).where(User.email == args.email.lower())): raise SystemExit('User already exists')
        db.add(User(email=args.email.lower(), password_hash=hash_password(password), role='administrator', is_active=True)); db.commit()
    finally: db.close()
    print('Administrator created.')
if __name__ == '__main__': main()
