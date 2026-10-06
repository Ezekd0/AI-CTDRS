from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ConfigDict

NetworkStatus = Literal['excellent', 'good', 'degraded', 'poor', 'unavailable', 'normal', 'suspicious', 'anomalous']


class BrowserNetworkInfo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    effective_type: str | None = Field(None, max_length=50)
    rtt_ms: float | None = Field(None, ge=0, allow_inf_nan=False)
    downlink_mbps: float | None = Field(None, ge=0, allow_inf_nan=False)
    latency_samples_ms: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(default_factory=list, max_length=5)
    failed_http_requests: int = Field(0, ge=0, le=5)


class NetworkReportCreateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    device_id: str = Field('browser', min_length=1, max_length=128)
    connection_type: str | None = Field(None, max_length=50)
    connectivity_status: str | None = Field(None, max_length=50)
    internet_status: str | None = Field(None, max_length=50)
    network_transport: str | None = Field(None, max_length=80)
    local_ip: None = None
    public_ip: None = None
    dns_status: str | None = Field(None, max_length=50)
    dns_servers: None = None
    carrier: None = None
    signal_strength: None = None
    latency_ms: float | None = Field(None, ge=0, allow_inf_nan=False)
    packet_loss_pct: None = None
    download_speed_mbps: float | None = Field(None, ge=0, allow_inf_nan=False)
    upload_speed_mbps: float | None = Field(None, ge=0, allow_inf_nan=False)
    overall_status: NetworkStatus | None = None
    metadata: BrowserNetworkInfo | None = None
    created_at: datetime | None = None


class NetworkReportResponse(BaseModel):
    id: str
    user_id: str | None
    device_id: str
    connection_type: str | None = None
    connectivity_status: str | None = None
    internet_status: str | None = None
    network_transport: str | None = None
    local_ip: str | None = None
    public_ip: str | None = None
    dns_status: str | None = None
    dns_servers: list[str] | None = None
    carrier: str | None = None
    signal_strength: str | None = None
    latency_ms: float | None = Field(None, ge=0, allow_inf_nan=False)
    packet_loss_pct: float | None = None
    download_speed_mbps: float | None = Field(None, ge=0, allow_inf_nan=False)
    upload_speed_mbps: float | None = Field(None, ge=0, allow_inf_nan=False)
    overall_status: NetworkStatus
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class NetworkReportListResponse(BaseModel):
    items: list[NetworkReportResponse]
    count: int
