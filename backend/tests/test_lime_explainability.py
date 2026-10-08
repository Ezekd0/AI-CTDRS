import numpy as np

from ml.explainability.lime_explainer import explain_local, MAX_BACKGROUND_ROWS, NUM_SAMPLES
from ml.training.models import RandomForestModel


def test_real_lime_is_bounded_and_cached(monkeypatch):
    from lime import lime_tabular
    original = lime_tabular.LimeTabularExplainer
    sizes = []
    def record(data, **kwargs):
        sizes.append(len(data))
        return original(data, **kwargs)
    monkeypatch.setattr(lime_tabular, 'LimeTabularExplainer', record)
    rng = np.random.default_rng(42)
    background = rng.normal(size=(1000, 3)).astype(np.float32)
    model = RandomForestModel(2, 42, {'n_estimators': 20})
    model.estimator.n_jobs = 1
    model.fit(background, (background[:, 0] > 0).astype(int))
    names = ['bytes', 'src_bytes', 'other']
    row = background[:1]
    result = explain_local(model, row, names, ['normal', 'ATTACK'], background)
    cached = model._local_lime_explainer[1]
    assert cached.discretizer.to_discretize == [0, 1, 2]
    assert result['num_samples'] == NUM_SAMPLES
    assert result['surrogate_score'] > 0
    assert any(abs(r['lime_contribution']) > 0 for r in result['features'])
    again = explain_local(model, row, names, ['normal', 'ATTACK'], background)
    assert model._local_lime_explainer[1] is cached
    assert again == result
    assert sizes == [MAX_BACKGROUND_ROWS]
