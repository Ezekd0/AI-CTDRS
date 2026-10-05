from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def new_id() -> str:
    return str(uuid.uuid4())

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

class User(TimestampMixin, Base):
    __tablename__ = 'users'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(50), default='analyst', nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    detections: Mapped[list['Detection']] = relationship(back_populates='user')
    alerts: Mapped[list['Alert']] = relationship(back_populates='user', foreign_keys='Alert.user_id')
    response_actions: Mapped[list['ResponseAction']] = relationship(back_populates='user')
    reports: Mapped[list['Report']] = relationship(back_populates='user')

class Dataset(TimestampMixin, Base):
    __tablename__ = 'datasets'
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    models: Mapped[list['Model']] = relationship(back_populates='dataset', cascade='all, delete-orphan')
    detections: Mapped[list['Detection']] = relationship(back_populates='dataset')

class Model(TimestampMixin, Base):
    __tablename__ = 'models'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_key: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    dataset: Mapped[Dataset] = relationship(back_populates='models')
    detections: Mapped[list['Detection']] = relationship(back_populates='model')
    __table_args__ = (Index('ix_models_dataset_task_name', 'dataset_id', 'task', 'name', unique=True),)

class Detection(TimestampMixin, Base):
    __tablename__ = 'detections'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey('datasets.id', ondelete='RESTRICT'), index=True, nullable=False)
    model_id: Mapped[str] = mapped_column(ForeignKey('models.id', ondelete='RESTRICT'), index=True, nullable=False)
    prediction: Mapped[str] = mapped_column(String(255), nullable=False)
    prediction_index: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    raw_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    original_feature_values: Mapped[list | None] = mapped_column(JSON)
    transformed_features: Mapped[list | None] = mapped_column(JSON)
    feature_names: Mapped[list | None] = mapped_column(JSON)
    explanation_status: Mapped[dict | None] = mapped_column(JSON)
    user: Mapped[User | None] = relationship(back_populates='detections')
    dataset: Mapped[Dataset] = relationship(back_populates='detections')
    model: Mapped[Model] = relationship(back_populates='detections')
    alerts: Mapped[list['Alert']] = relationship(back_populates='detection', cascade='all, delete-orphan')
    explanations: Mapped[list['Explanation']] = relationship(back_populates='detection', cascade='all, delete-orphan')
    response_actions: Mapped[list['ResponseAction']] = relationship(back_populates='detection', cascade='all, delete-orphan')
    reports: Mapped[list['Report']] = relationship(back_populates='detection')
    __table_args__ = (Index('ix_detections_created_at', 'created_at'), Index('ix_detections_prediction', 'prediction'))

class Alert(TimestampMixin, Base):
    __tablename__ = 'alerts'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    detection_id: Mapped[str] = mapped_column(ForeignKey('detections.id', ondelete='CASCADE'), index=True, nullable=False)
    user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), index=True)
    assigned_analyst_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), index=True)
    severity: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default='new', nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    detection: Mapped[Detection] = relationship(back_populates='alerts')
    user: Mapped[User | None] = relationship(back_populates='alerts', foreign_keys=[user_id])
    assigned_analyst: Mapped[User | None] = relationship(foreign_keys=[assigned_analyst_id])
    audit_logs: Mapped[list['AlertAuditLog']] = relationship(back_populates='alert', cascade='all, delete-orphan')

class AlertAuditLog(TimestampMixin, Base):
    __tablename__ = 'alert_audit_logs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    alert_id: Mapped[str] = mapped_column(ForeignKey('alerts.id', ondelete='CASCADE'), index=True, nullable=False)
    user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), index=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    previous_status: Mapped[str | None] = mapped_column(String(30))
    new_status: Mapped[str | None] = mapped_column(String(30))
    details: Mapped[dict | None] = mapped_column(JSON)
    alert: Mapped[Alert] = relationship(back_populates='audit_logs')
    user: Mapped[User | None] = relationship()

class Explanation(TimestampMixin, Base):
    __tablename__ = 'explanations'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    detection_id: Mapped[str] = mapped_column(ForeignKey('detections.id', ondelete='CASCADE'), index=True, nullable=False)
    method: Mapped[str] = mapped_column(String(30), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    detection: Mapped[Detection] = relationship(back_populates='explanations')
    __table_args__ = (Index('ix_explanations_detection_method', 'detection_id', 'method', unique=True),)

class ResponseAction(TimestampMixin, Base):
    __tablename__ = 'response_actions'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    detection_id: Mapped[str] = mapped_column(ForeignKey('detections.id', ondelete='CASCADE'), index=True, nullable=False)
    user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), index=True)
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default='pending', nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default='No reason provided')
    detection: Mapped[Detection] = relationship(back_populates='response_actions')
    user: Mapped[User | None] = relationship(back_populates='response_actions')

class Report(TimestampMixin, Base):
    __tablename__ = 'reports'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), index=True)
    detection_id: Mapped[str | None] = mapped_column(ForeignKey('detections.id', ondelete='SET NULL'), index=True)
    report_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default='generated', nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(500))
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    user: Mapped[User | None] = relationship(back_populates='reports')
    detection: Mapped[Detection | None] = relationship(back_populates='reports')

class RevokedToken(Base):
    __tablename__ = 'revoked_tokens'
    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
