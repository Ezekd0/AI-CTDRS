"""Add optional full name without changing existing accounts."""
from alembic import op
import sqlalchemy as sa

revision = "0005_user_full_name"
down_revision = "0004_alert_workflow_audit"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("full_name", sa.String(200), nullable=True))


def downgrade():
    op.drop_column("users", "full_name")
