from fastapi.testclient import TestClient

from wearwise_ai.app import create_app


def test_liveness_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_manifest_uses_correlation_id() -> None:
    client = TestClient(create_app())

    response = client.get(
        "/internal/v1/service/manifest",
        headers={"X-Request-ID": "test-correlation-id"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-correlation-id"
    assert response.json()["meta"]["request_id"] == "test-correlation-id"
    assert response.json()["data"]["service"] == "wearwise-ai"
