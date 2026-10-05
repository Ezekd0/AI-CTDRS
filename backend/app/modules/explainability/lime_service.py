from __future__ import annotations

from app.db.session import get_engine
from app.db.models.models import Detection
from app.modules.explainability.service import generate_lime_for_detection
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload


def explain_detection_lime(detection_id: str):
    with Session(get_engine()) as db:
        detection = db.execute(select(Detection).options(joinedload(Detection.model)).where(Detection.id == detection_id)).scalar_one_or_none()
        if detection is None:
            raise FileNotFoundError(detection_id)
        result = generate_lime_for_detection(db, detection)
        db.commit()
        return result
