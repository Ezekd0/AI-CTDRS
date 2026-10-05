"""Add alert assignment, controlled statuses and audit log."""
from alembic import op
import sqlalchemy as sa

revision = "0004_alert_workflow_audit"
down_revision = "0003_response_reason"
branch_labels = None
depends_on = None

def upgrade():
    op.execute("UPDATE alerts SET status = 'new' WHERE status = 'open'")
    op.add_column("alerts", sa.Column("assigned_analyst_id", sa.String(length=36), nullable=True))
    op.create_index("ix_alerts_assigned_analyst_id", "alerts", ["assigned_analyst_id"], unique=False)
    op.create_foreign_key("fk_alerts_assigned_analyst_id_users", "alerts", "users", ["assigned_analyst_id"], ["id"], ondelete="SET NULL")
    op.create_table("alert_audit_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("alert_id", sa.String(length=36), sa.ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("previous_status", sa.String(length=30), nullable=True),
        sa.Column("new_status", sa.String(length=30), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_alert_audit_logs_alert_id", "alert_audit_logs", ["alert_id"])
    op.create_index("ix_alert_audit_logs_user_id", "alert_audit_logs", ["user_id"])

def downgrade():
    op.drop_index("ix_alert_audit_logs_user_id", table_name="alert_audit_logs")
    op.drop_index("ix_alert_audit_logs_alert_id", table_name="alert_audit_logs")
    op.drop_table("alert_audit_logs")
    op.drop_constraint("fk_alerts_assigned_analyst_id_users", "alerts", type_="foreignkey")
    op.drop_index("ix_alerts_assigned_analyst_id", table_name="alerts")
    op.drop_column("alerts", "assigned_analyst_id")
