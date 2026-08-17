"""Tests for src/serve.py API-key verification."""

import pytest
from fastapi.testclient import TestClient

from src.serve import app

API_KEY = "test-key-for-ci"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SMARTLITE_API_KEY", API_KEY)
    return TestClient(app)


class TestVerifyApiKey:
    def test_valid_key_passes_auth(self, client):
        response = client.get("/model/info", headers={"X-API-Key": API_KEY})

        assert response.status_code != 403

    def test_missing_key_returns_403(self, client):
        assert client.get("/model/info").status_code == 403

    def test_wrong_key_returns_403(self, client):
        response = client.get("/model/info", headers={"X-API-Key": "nope"})

        assert response.status_code == 403

    def test_non_ascii_key_returns_403_not_500(self, client):
        """A non-ASCII header must be a clean auth failure, not a TypeError.

        The value is pre-encoded because httpx refuses non-ASCII str headers;
        on the wire this is exactly what a non-ASCII key looks like.
        """
        response = client.get(
            "/model/info", headers={"X-API-Key": "clé-münchen-🔑".encode()}
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "Invalid or missing API key"

    def test_non_ascii_key_raises_403_directly(self, monkeypatch):
        """secrets.compare_digest would raise TypeError on these as str."""
        import asyncio

        from fastapi import HTTPException

        from src.serve import verify_api_key

        monkeypatch.setenv("SMARTLITE_API_KEY", API_KEY)

        with pytest.raises(HTTPException) as exc:
            asyncio.run(verify_api_key(key="clé-münchen"))

        assert exc.value.status_code == 403

    def test_non_ascii_server_key_still_matches_itself(self, monkeypatch):
        import asyncio

        from src.serve import verify_api_key

        monkeypatch.setenv("SMARTLITE_API_KEY", "clé-münchen")

        assert asyncio.run(verify_api_key(key="clé-münchen")) is None

    def test_unset_server_key_returns_403(self, client, monkeypatch):
        monkeypatch.delenv("SMARTLITE_API_KEY")

        response = client.get("/model/info", headers={"X-API-Key": API_KEY})

        assert response.status_code == 403


class TestPublicEndpoints:
    def test_health_needs_no_key(self, client):
        assert client.get("/health").status_code == 200
