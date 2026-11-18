import pytest
from langsmith import tracing_context


@pytest.fixture(autouse=True)
def disable_external_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    with tracing_context(enabled=False):
        yield
