"""Shared fixtures for tests/tools/.

Strips system proxy env vars (ALL_PROXY etc.) before each test so respx-mocked
httpx clients don't try to set up a SOCKS transport (which would require the
optional `socksio` package). Tool wrappers themselves keep the default
`trust_env=True` httpx behavior so production runs respect operator-configured
proxies.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _strip_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for proxy_var in (
        "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
        "all_proxy", "http_proxy", "https_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)
