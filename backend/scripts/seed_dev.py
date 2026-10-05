"""Seed only non-sensitive development configuration. No users or incidents are created."""
from sqlalchemy import select
from app.db.session import get_engine
from app.db.models import Base, Dataset, Model
from app.db.models.models import new_id
from sqlalchemy.orm import Session

DATASETS = [
    ('cicids2017', 'CICIDS2017'),
    ('cicids2018', 'CICIDS2018'),
    ('nsl-kdd', 'NSL-KDD'),
]

def main():
    Base.metadata.create_all(get_engine())
    with Session(get_engine()) as db:
        for dataset_id, name in DATASETS:
            dataset = db.get(Dataset, dataset_id)
            if dataset is None:
                dataset = Dataset(id=dataset_id, name=name, description='Development dataset configuration only.')
                db.add(dataset)
                db.flush()
            for task in ('binary', 'multiclass'):
                for model_name in ('random_forest', 'xgboost'):
                    exists = db.scalar(select(Model).where(Model.dataset_id==dataset_id, Model.task==task, Model.name==model_name))
                    if exists is None:
                        db.add(Model(dataset_id=dataset_id, task=task, name=model_name, artifact_key=f'{dataset_id}/{task}/{model_name}', is_active=True))
        db.commit()

if __name__ == '__main__': main()
