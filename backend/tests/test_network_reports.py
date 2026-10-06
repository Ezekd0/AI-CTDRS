import os

import pytest
from pydantic import ValidationError
from app.schemas.network import NetworkReportCreateRequest

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_access_token
from app.db.models import Base, User, NetworkReport
from app.db.session import get_db, get_engine
from app.main import app


def _setup(tmp_path):
    db_url = os.environ.get("TEST_NETWORK_DATABASE_URL") or f"sqlite:///{tmp_path / 'network_reports.db'}"
    os.environ['DATABASE_URL'] = db_url
    get_settings.cache_clear()
    get_engine.cache_clear()
    Base.metadata.create_all(get_engine())

    with Session(get_engine()) as db:
        db.add_all([
            User(id='user-1', email='alice@example.com', password_hash='x', role='viewer', is_active=True),
            User(id='user-2', email='bob@example.com', password_hash='x', role='viewer', is_active=True),
            User(id='admin-1', email='admin@example.com', password_hash='x', role='administrator', is_active=True),
        ])
        db.commit()

    def override_db():
        with Session(get_engine()) as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def _token(user_id: str) -> str:
    with Session(get_engine()) as db:
        user = db.get(User, user_id)
        return create_access_token(user)[0]


def test_network_reports_are_user_scoped_and_admin_visible(tmp_path):
    client = _setup(tmp_path)
    try:
        alice = {'Authorization': f'Bearer {_token("user-1")}'}
        bob = {'Authorization': f'Bearer {_token("user-2")}'}
        admin = {'Authorization': f'Bearer {_token("admin-1")}'}

        payload = {
            'device_id': 'browser',
            'connection_type': 'wifi',
            'connectivity_status': 'connected',
            'internet_status': 'reachable',
            'network_transport': 'wifi',
            'latency_ms': 18.4,
            'download_speed_mbps': 54.7,
            'upload_speed_mbps': 12.3,
            'dns_status': 'Backend HTTP reachable; DNS may be cached',
            'metadata': {'effective_type': '4g', 'rtt_ms': 50, 'downlink_mbps': 10, 'latency_samples_ms': [18.4], 'failed_http_requests': 0},
            'overall_status': 'normal',
            'created_at': '2026-10-05T12:00:00Z'
        }

        response = client.post('/api/network/reports', headers=alice, json=payload)
        assert response.status_code == 201, response.text
        report_id = response.json()['id']
        assert response.json()['user_id'] == 'user-1'
        assert response.json()['connection_type'] == 'wifi'
        with Session(get_engine()) as persisted:
            stored = persisted.get(NetworkReport, report_id)
            assert stored.latency_ms == 18.4
            assert stored.user_id == 'user-1'
            assert stored.download_speed_mbps == 54.7

        latest = client.get('/api/network/reports/latest', headers=alice)
        assert latest.status_code == 200, latest.text
        assert latest.json()['id'] == report_id

        list_for_bob = client.get('/api/network/reports', headers=bob)
        assert list_for_bob.status_code == 200, list_for_bob.text
        assert list_for_bob.json()['items'] == []

        assert client.get('/api/network/reports').status_code == 401
        assert client.get('/api/network/reports?user_id=user-1', headers=bob).json()['items'] == []
        assert client.get(f'/api/network/reports/{report_id}', headers=bob).status_code == 403
        assert client.get(f'/api/network/reports/{report_id}', headers=admin).status_code == 200
        assert response.json()['overall_status'] == 'excellent'
        assert client.get('/api/network/latency').status_code == 401
        assert client.get('/api/network/latency', headers=alice).json()['status'] == 'reachable'
        probe = client.get('/api/network/probe?size_bytes=1000000', headers=alice)
        assert len(probe.content) == 1000000
        assert probe.headers['content-type'] == 'application/octet-stream'
        assert client.get('/api/network/probe?size_bytes=2000001', headers=alice).status_code == 422
        upload = client.post('/api/network/probe', headers={**alice, 'Content-Type': 'application/octet-stream'}, content=b'x' * 1000000)
        assert upload.json()['bytes_received'] == 1000000
        assert client.post('/api/network/probe', headers={**alice, 'Content-Type': 'application/octet-stream'}, content=b'x' * 2000001).status_code == 413
        chunked = client.post('/api/network/probe', headers={**alice, 'Content-Type': 'application/octet-stream'}, content=iter([b'x' * 1000000, b'x' * 1000001]))
        assert chunked.status_code == 413
        assert client.post('/api/network/reports', headers=alice, json={**payload, 'metadata': {'browser_contents': 'private'}}).status_code == 422
        assert client.post('/api/network/reports', headers=alice, json={**payload, 'latency_ms': -1}).status_code == 422
        assert client.post('/api/network/reports', headers=alice, json={**payload, 'packet_loss_pct': 0}).status_code == 422
        for _ in range(20):
            limited = client.get('/api/network/latency', headers=bob)
        assert limited.status_code == 200
        assert client.get('/api/network/latency', headers=bob).status_code == 429
        admin_list = client.get('/api/network/reports', headers=admin)
        assert admin_list.status_code == 200, admin_list.text
        assert admin_list.json()['count'] >= 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize('sample', [-1, float('nan'), float('inf'), float('-inf')])
def test_latency_samples_reject_negative_and_non_finite_values(sample):
    with pytest.raises(ValidationError) as exc:
        NetworkReportCreateRequest(metadata={'latency_samples_ms': [0, sample, 10]})
    assert exc.value.errors()[0]['loc'] == ('metadata', 'latency_samples_ms', 1)


@pytest.mark.parametrize('samples', [[], [0], [0, 0.25, 10, 100.5, 1000]])
def test_latency_samples_accept_non_negative_finite_values(samples):
    report = NetworkReportCreateRequest(metadata={'latency_samples_ms': samples})
    assert report.metadata.latency_samples_ms == samples
