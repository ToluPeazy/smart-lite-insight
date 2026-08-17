"""Tests for dashboard/auth.py — the dashboard shared-secret gate."""

from dashboard.auth import ENV_VAR, expected_password, is_authorised


class TestIsAuthorised:
    def test_correct_password_accepted(self):
        assert is_authorised("s3cret", "s3cret") is True

    def test_wrong_password_rejected(self):
        assert is_authorised("nope", "s3cret") is False

    def test_unset_secret_locks_rather_than_opens(self):
        assert is_authorised("anything", None) is False
        assert is_authorised("anything", "") is False

    def test_empty_supplied_rejected(self):
        assert is_authorised("", "s3cret") is False
        assert is_authorised(None, "s3cret") is False

    def test_non_ascii_password_does_not_raise(self):
        assert is_authorised("pässwörd", "pässwörd") is True
        assert is_authorised("pässwörd", "password") is False


class TestExpectedPassword:
    def test_reads_env_at_call_time(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert expected_password() is None

        monkeypatch.setenv(ENV_VAR, "s3cret")
        assert expected_password() == "s3cret"

    def test_empty_env_var_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "")
        assert expected_password() is None
