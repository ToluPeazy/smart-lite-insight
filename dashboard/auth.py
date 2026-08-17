"""Shared-secret gate for the Streamlit dashboard.

The dashboard reads SQLite directly and drives the in-process LLM agent, so it
never passes through the API's `X-API-Key` check. Anything that can reach port
8501 can read the energy data. This module is the minimum viable barrier: a
single shared secret in `SMARTLITE_DASHBOARD_PASSWORD`.

It is deliberately small and is *not* a substitute for a real auth layer. Put
Cloudflare Access (or equivalent) in front of the dashboard, or keep 8501 on
the LAN — see README.md.
"""

import hmac
import os

ENV_VAR = "SMARTLITE_DASHBOARD_PASSWORD"


def is_authorised(supplied: str | None, expected: str | None) -> bool:
    """Constant-time check of a supplied password against the expected one.

    Returns False when either side is empty, so an unset secret locks the
    dashboard rather than opening it.
    """
    if not expected or not supplied:
        return False

    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def expected_password() -> str | None:
    """Read the configured dashboard secret (at call time, not import time)."""
    return os.getenv(ENV_VAR) or None


def require_password() -> bool:
    """Gate the app on the shared secret. Returns True when unlocked.

    Renders a password prompt and returns False otherwise, so callers can
    `if not require_password(): return` before drawing anything.
    """
    import streamlit as st

    expected = expected_password()

    if st.session_state.get("dashboard_unlocked"):
        return True

    st.title("⚡ Smart-Lite Insight")

    if expected is None:
        st.error(
            f"Dashboard locked: {ENV_VAR} is not set. "
            "Set it in the environment (see .env.example) and restart."
        )
        return False

    supplied = st.text_input("Dashboard password", type="password")

    if not supplied:
        st.info("Enter the dashboard password to continue.")
        return False

    if not is_authorised(supplied, expected):
        st.error("Incorrect password.")
        return False

    st.session_state["dashboard_unlocked"] = True
    return True
