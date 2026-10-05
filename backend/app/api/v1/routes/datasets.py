from fastapi import APIRouter, Depends
from app.core.security import require_roles
from ml.training.datasets import DATASETS

router = APIRouter(prefix='/datasets', tags=['Datasets'])

@router.get('')
def datasets(user=Depends(require_roles('administrator', 'analyst', 'viewer'))):
    return {'items': [
        {
            'id': spec.key,
            'name': spec.display_name,
            'artifact_slug': spec.slug,
            'raw_dir': spec.raw_dir,
            'preprocess_command': spec.preprocess_command,
        }
        for spec in DATASETS.values()
    ]}
