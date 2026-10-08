"""Small per-process caches for immutable deployed inference artifacts.

Deployments restart workers when artifacts change. Keys include the absolute root
so different installations/test bundles never share a loaded model.
"""
from functools import lru_cache
from pathlib import Path

import joblib

from ml.training.artifacts import load_bundle
from ml.training.datasets import resolve_dataset


@lru_cache(maxsize=1)
def cached_bundle(root: Path, dataset: str, task: str, model: str):
    bundle = load_bundle(root, dataset, task, model)
    if model == 'random_forest':
        # A small Render instance should not fan each inference out to every CPU.
        bundle.model.estimator.n_jobs = 1
    return bundle


@lru_cache(maxsize=1)
def cached_preprocessor(root: Path, dataset: str, task: str, model: str):
    spec = resolve_dataset(dataset)
    directory = (root / spec.slug / task / model).resolve()
    if root != directory and root not in directory.parents:
        raise ValueError('Artifact path is outside configured root')
    path = directory / 'preprocessor.joblib'
    if not path.exists():
        raise FileNotFoundError('Saved preprocessing artifact not found')
    saved = joblib.load(path)
    if isinstance(saved, dict):
        import importlib
        module = importlib.import_module(spec.preprocessor_module)
        return getattr(module, spec.preprocessor_class).load(path)
    return saved
