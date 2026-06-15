"""
framework/dashboard_models.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic request models used by the live dashboard API.

These models are only for HTTP request/response schemas (FastAPI endpoints).
All other domain data structures are now in framework/types.py.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProviderRequest(BaseModel):
    """Provider credentials for agent API endpoint."""

    id: str = ""
    label: str = ""
    base_url: str
    model: str
    api_key: str = ""
    source: str = ""
    enabled: bool = True


class RunRequest(BaseModel):
    """Request to start a benchmark run from the dashboard."""

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    level: str = ""
    all_levels: bool = False
    capture_preview_screenshot: bool = True
    save_result_bundle: bool = False
    save_container_snapshot: bool = False
    providers: list[ProviderRequest] = []


class ChatRequest(BaseModel):
    """Request for the assistant chatbot endpoint."""

    messages: list[dict]
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    active_run_id: str = ""
    page: str = "dashboard"
    page_context: str = ""
    level: str = ""
    all_levels: bool = False
    capture_preview_screenshot: bool = True
    save_result_bundle: bool = False
    save_container_snapshot: bool = False
    parallel_compare: bool = False
    providers: list[ProviderRequest] = []
    allow_control: bool = True


class SaveLevelRequest(BaseModel):
    """Request to save a level YAML."""

    filename: str
    content: str


class TestProviderRequest(BaseModel):
    """Request to test one or more endpoint URLs."""

    urls: list[str] = []


class SaveProvidersRequest(BaseModel):
    """Request to replace the stored provider list."""

    providers: list[ProviderRequest] = []
