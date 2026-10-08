"""Enforce deadlines using a real spawned process, without touching DB sessions."""
import time

import pytest

from app.modules.explainability.worker import ExplanationWorker


def serve_test(connection):
    while True:
        method, data = connection.recv()
        if method == 'slow':
            time.sleep(30)
        connection.send((True, data))


def test_timeout_kills_worker_and_next_request_recovers():
    worker = ExplanationWorker(target=serve_test)
    try:
        assert worker.run('fast', {'value': 1}, timeout=5) == {'value': 1}
        process = worker._process
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            worker.run('slow', {}, timeout=0.1)
        assert time.monotonic() - started < 1
        assert worker._process is None
        assert process._closed
        assert worker.run('fast', {'value': 2}, timeout=5) == {'value': 2}
        process = worker._process
        assert worker.run('fast', {}, timeout=5) == {}
        assert worker._process is process
    finally:
        worker.close()


def test_busy_worker_does_not_queue_more_computation():
    worker = ExplanationWorker()
    worker._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='busy'):
            worker.run('shap', {}, timeout=10)
        assert worker._process is None
    finally:
        worker._lock.release()


def test_exhausted_budget_does_not_start_worker():
    worker = ExplanationWorker()
    with pytest.raises(TimeoutError, match='budget exhausted'):
        worker.run('lime', {}, timeout=0)
    assert worker._process is None
