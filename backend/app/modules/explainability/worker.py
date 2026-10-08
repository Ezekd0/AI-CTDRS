"""One reusable, killable explanation process per API worker, with no task queue.

Only plain feature data crosses the pipe. Database sessions stay in the parent.
Spawn avoids forking threaded ASGI/BLAS state. A timeout kills the computation,
including cold imports; the next request starts a fresh worker.
"""
from __future__ import annotations

import atexit
import multiprocessing
import os
import logging
from pathlib import Path
from threading import Lock, Thread
from time import monotonic


def _calculate(method, data):
    import numpy as np
    from app.modules.prediction.runtime import cached_bundle
    bundle = cached_bundle(Path(data['root']), data['dataset'], data['task'], data['model'])
    if method == 'warmup':
        from ml.explainability.shap_explainer import _cached_tree_explainer, _tree_estimator
        from ml.explainability.lime_explainer import _explainer
        _cached_tree_explainer(_tree_estimator(bundle.model, data['model']))
        if bundle.lime_background is not None:
            _explainer(bundle.model, bundle.lime_background, bundle.feature_names, bundle.class_names)
        return {'ready': True}
    X = np.asarray(data['features'], dtype=np.float32).reshape(1, -1)
    if method == 'shap':
        from ml.explainability.shap_explainer import explain_local
        return explain_local(bundle.model, data['model'], X, data['names'], bundle.class_names, data['originals'])
    if method == 'lime':
        from ml.explainability.lime_explainer import explain_local
        if bundle.lime_background is None:
            raise RuntimeError('LIME background artifact is missing for this model run')
        return explain_local(bundle.model, X, data['names'], bundle.class_names,
                             training_data=bundle.lime_background, original_values=data['originals'])
    raise ValueError('Unsupported explanation method')


def _serve(connection):
    # Prevent nested BLAS pools competing with the API on small Render instances.
    # Set these in the child before importing numpy/scipy, avoiding cold-start
    # thread creation as well as limiting libraries that are imported later.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from app.modules.prediction.runtime import cached_bundle  # loads numeric libraries before limiting them
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        try:
            while True:
                method, data = connection.recv()
                try:
                    connection.send((True, _calculate(method, data)))
                except Exception as exc:
                    connection.send((False, f'{type(exc).__name__}: {exc}'))
        except (EOFError, BrokenPipeError):
            pass
        finally:
            connection.close()


class ExplanationWorker:
    def __init__(self, target=_serve):
        self._lock = Lock()
        self._process = None
        self._connection = None
        self._target = target
        self._warming = False

    def start_warmup(self, root):
        """Cold imports on 0.1 CPU can exceed a request budget; prepare once.

        Initialization has a separate three-minute cap and never blocks startup
        or predictions. It builds explainers, not fabricated sample explanations.
        """
        if self._warming or self._process is not None:
            return
        if not (root / 'nsl_kdd/binary/random_forest/metadata.json').exists():
            return
        self._warming = True
        def warm():
            try:
                self.run('warmup', {'root': str(root), 'dataset': 'nsl-kdd',
                                   'task': 'binary', 'model': 'random_forest'}, timeout=180)
            except Exception:
                logging.getLogger(__name__).exception('Explanation warm-up unavailable')
            finally:
                self._warming = False
        Thread(target=warm, name='explanation-warmup', daemon=True).start()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.terminate()
            self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.2)
            self._process.close()
            self._process = None

    def run(self, method, data, timeout=20.0):
        if timeout <= 0:
            raise TimeoutError('Explanation request budget exhausted')
        if not self._lock.acquire(blocking=False):
            if self._warming:
                raise RuntimeError('Explanation worker warming up; retry explanation later')
            raise RuntimeError('Explanation worker busy; retry explanation later')
        deadline = monotonic() + timeout
        try:
            if self._process is None or not self._process.is_alive():
                self.close()
                context = multiprocessing.get_context('spawn')
                self._connection, child = context.Pipe()
                self._process = context.Process(target=self._target, args=(child,), daemon=True)
                self._process.start()
                child.close()
            self._connection.send((method, data))
            if not self._connection.poll(max(0, deadline - monotonic())):
                raise TimeoutError(f'{method.upper()} exceeded its server-side time limit')
            ok, result = self._connection.recv()
            if not ok:
                raise RuntimeError(result)
            return result
        except (TimeoutError, EOFError, BrokenPipeError, OSError):
            self.close()
            raise
        finally:
            self._lock.release()


worker = ExplanationWorker()
atexit.register(worker.close)
