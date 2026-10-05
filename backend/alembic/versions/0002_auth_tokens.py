"""add revoked tokens for JWT logout invalidation

Revision ID: 0002_auth_tokens
Revises: 0001_initial_schema
"""
from alembic import op
import sqlalchemy as sa
revision='0002_auth_tokens'; down_revision='0001_initial_schema'; branch_labels=None; depends_on=None

def upgrade():
    op.create_table('revoked_tokens',
        sa.Column('jti', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('jti'))
    op.create_index('ix_revoked_tokens_expires_at', 'revoked_tokens', ['expires_at'])

def downgrade():
    op.drop_index('ix_revoked_tokens_expires_at', table_name='revoked_tokens')
    op.drop_table('revoked_tokens')
