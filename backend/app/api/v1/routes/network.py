from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.models.models import NetworkReport, User
from app.db.session import get_db
from app.schemas.network import NetworkReportCreateRequest, NetworkReportListResponse, NetworkReportResponse

router = APIRouter(prefix='/network', tags=['Network Monitor'])


def _assess_status(payload: NetworkReportCreateRequest) -> str:
    if payload.internet_status != 'reachable':
        return 'unavailable'
    latency, download, upload = payload.latency_ms, payload.download_speed_mbps, payload.upload_speed_mbps
    if latency is None or download is None or upload is None:
        return 'unavailable'
    if latency >= 600 or download < 1 or upload < 0.5:
        return 'poor'
    if latency >= 250 or download < 3 or upload < 1:
        return 'degraded'
    if latency < 100 and download >= 25 and upload >= 5:
        return 'excellent'
    return 'good'


# Per-worker limiter; payload limits apply on every worker. No uploaded contents are retained.
_probe_calls = defaultdict(deque)
_probe_lock = Lock()


def _limit_probe(user: User = Depends(get_current_user)):
    now = time.monotonic()
    with _probe_lock:
        for key in list(_probe_calls):
            if not _probe_calls[key] or _probe_calls[key][-1] <= now - 60:
                del _probe_calls[key]
        calls = _probe_calls[user.id]
        while calls and calls[0] <= now - 60:
            calls.popleft()
        if len(calls) >= 20:
            raise HTTPException(429, 'Network test rate limit reached; wait one minute')
        calls.append(now)
    return user


def _serialize(report: NetworkReport) -> NetworkReportResponse:
    return NetworkReportResponse(
        id=report.id,
        user_id=report.user_id,
        device_id=report.device_id,
        connection_type=report.connection_type,
        connectivity_status=report.connectivity_status,
        internet_status=report.internet_status,
        network_transport=report.network_transport,
        local_ip=report.local_ip,
        public_ip=report.public_ip,
        dns_status=report.dns_status,
        dns_servers=report.dns_servers or [],
        carrier=report.carrier,
        signal_strength=report.signal_strength,
        latency_ms=report.latency_ms,
        packet_loss_pct=report.packet_loss_pct,
        download_speed_mbps=report.download_speed_mbps,
        upload_speed_mbps=report.upload_speed_mbps,
        overall_status=report.overall_status,
        metadata=report.metadata_json or {},
        created_at=report.created_at,
    )


@router.get('/reports', response_model=NetworkReportListResponse)
def list_reports(
    user_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    filters = []
    if current_user.role != 'administrator':
        filters.append(NetworkReport.user_id == current_user.id)
    elif user_id is not None:
        filters.append(NetworkReport.user_id == user_id)

    rows = db.scalars(
        select(NetworkReport).where(*filters).order_by(NetworkReport.created_at.desc()).limit(limit)
    ).all()
    return {'items': [_serialize(r) for r in rows], 'count': len(rows)}


@router.get('/reports/latest', response_model=NetworkReportResponse)
def latest_report(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    filters = [NetworkReport.user_id == current_user.id]
    row = db.scalar(
        select(NetworkReport).where(*filters).order_by(NetworkReport.created_at.desc()).limit(1)
    )
    if row is None:
        raise HTTPException(status_code=404, detail='No network reports found for this account')
    return _serialize(row)


@router.get('/reports/{report_id}', response_model=NetworkReportResponse)
def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = db.get(NetworkReport, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail='Network report not found')
    if current_user.role != 'administrator' and report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail='Access denied')
    return _serialize(report)


@router.get('/latency')
def latency_probe(current_user: User = Depends(_limit_probe)):
    return {'status': 'reachable'}


@router.get('/probe')
def download_probe(
    size_bytes: int = Query(1_000_000, ge=64, le=2_000_000),
    current_user: User = Depends(_limit_probe),
):
    return Response(os.urandom(size_bytes), media_type='application/octet-stream',
                    headers={'Cache-Control': 'no-store, no-transform'})


@router.post('/probe')
async def upload_probe(request: Request, current_user: User = Depends(_limit_probe)):
    if request.headers.get('content-type', '').split(';')[0] != 'application/octet-stream':
        raise HTTPException(415, 'Use application/octet-stream')
    length = request.headers.get('content-length')
    if length and (not length.isdigit() or int(length) > 2_000_000):
        raise HTTPException(413, 'Maximum upload is 2000000 bytes')
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > 2_000_000:
            raise HTTPException(413, 'Maximum upload is 2000000 bytes')
    if not received:
        raise HTTPException(400, 'Upload is empty')
    return {'bytes_received': received, 'status': 'accepted'}


@router.post('/reports', response_model=NetworkReportResponse, status_code=201)
def create_report(
    body: NetworkReportCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    status_value = _assess_status(body)
    report = NetworkReport(
        user_id=current_user.id,
        device_id=body.device_id,
        connection_type=body.connection_type,
        connectivity_status=body.connectivity_status,
        internet_status=body.internet_status,
        network_transport=body.network_transport,
        local_ip=body.local_ip,
        public_ip=body.public_ip,
        dns_status=body.dns_status,
        dns_servers=body.dns_servers,
        carrier=body.carrier,
        signal_strength=body.signal_strength,
        latency_ms=body.latency_ms,
        packet_loss_pct=body.packet_loss_pct,
        download_speed_mbps=body.download_speed_mbps,
        upload_speed_mbps=body.upload_speed_mbps,
        overall_status=status_value,
        metadata_json=body.metadata.model_dump() if body.metadata else {},
        created_at=body.created_at or datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return _serialize(report)
