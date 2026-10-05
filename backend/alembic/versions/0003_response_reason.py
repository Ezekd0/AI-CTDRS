"""Add explicit response action reason field."""
from alembic import op
import sqlalchemy as sa

revision = "0003_response_reason"
down_revision = "0002_auth_tokens"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("response_actions", sa.Column("reason", sa.Text(), nullable=True))
    op.execute("UPDATE response_actions SET reason = 'No reason provided' WHERE reason IS NULL")
    op.alter_column("response_actions", "reason", nullable=False, server_default="No reason provided")

def downgrade():
    op.drop_column("response_actions", "reason")
