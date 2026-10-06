"""Add authenticated browser network reports."""
from alembic import op
import sqlalchemy as sa

revision = "0006_network_reports"
down_revision = "0005_user_full_name"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'network_reports',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('device_id', sa.String(length=128), nullable=False),
        sa.Column('connection_type', sa.String(length=50), nullable=True),
        sa.Column('connectivity_status', sa.String(length=50), nullable=True),
        sa.Column('internet_status', sa.String(length=50), nullable=True),
        sa.Column('network_transport', sa.String(length=80), nullable=True),
        sa.Column('local_ip', sa.String(length=100), nullable=True),
        sa.Column('public_ip', sa.String(length=100), nullable=True),
        sa.Column('dns_status', sa.String(length=50), nullable=True),
        sa.Column('dns_servers', sa.JSON(), nullable=True),
        sa.Column('carrier', sa.String(length=128), nullable=True),
        sa.Column('signal_strength', sa.String(length=80), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('packet_loss_pct', sa.Float(), nullable=True),
        sa.Column('download_speed_mbps', sa.Float(), nullable=True),
        sa.Column('upload_speed_mbps', sa.Float(), nullable=True),
        sa.Column('overall_status', sa.String(length=30), nullable=False, server_default='normal'),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_network_reports_user_id'), 'network_reports', ['user_id'], unique=False)
    op.create_index(op.f('ix_network_reports_device_id'), 'network_reports', ['device_id'], unique=False)
    op.create_index(op.f('ix_network_reports_created_at'), 'network_reports', ['created_at'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_network_reports_created_at'), table_name='network_reports')
    op.drop_index(op.f('ix_network_reports_device_id'), table_name='network_reports')
    op.drop_index(op.f('ix_network_reports_user_id'), table_name='network_reports')
    op.drop_table('network_reports')
