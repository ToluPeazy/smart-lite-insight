"""Streamlit Dashboard for Smart-Lite Insight.

Interactive energy monitoring dashboard with:
- Real-time consumption chart with anomaly overlay
- Anomaly details table
- Model info sidebar
- Date range selector

Usage:
    streamlit run dashboard/app.py
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page Config ──
st.set_page_config(
    page_title="Smart-Lite Insight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──
DB_PATH = "data/processed/energy.db"
MODELS_DIR = "models"


# ── Data Loading ──


@st.cache_data(ttl=60)
def load_data(site_id: str, start: str, end: str) -> pd.DataFrame:
    """Load energy readings from SQLite for a given time range."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            timestamp,
            global_active_power_kw,
            global_reactive_power_kw,
            voltage_v,
            global_intensity_a,
            sub_metering_1_wh,
            sub_metering_2_wh,
            sub_metering_3_wh
        FROM readings
        WHERE site_id = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        """,
        conn,
        params=(site_id, start, end),
        parse_dates=["timestamp"],
    )
    conn.close()
    return df


@st.cache_data(ttl=60)
def get_date_range(site_id: str) -> tuple[datetime, datetime]:
    """Get the min and max timestamps in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM readings WHERE site_id = ?",
        (site_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row[0] is None:
        now = datetime.now()
        return now - timedelta(days=7), now

    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


def load_model_info() -> dict | None:
    """Load model registry metadata."""
    registry_path = Path(MODELS_DIR) / "registry.json"
    if not registry_path.is_file():
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    if not registry.get("models"):
        return None

    return registry


def score_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """Score data for anomalies using the trained model."""
    try:
        from src.detect import AnomalyDetector
        from src.features import build_feature_matrix

        df_indexed = df.copy()
        df_indexed["timestamp"] = pd.to_datetime(df_indexed["timestamp"])
        df_indexed = df_indexed.set_index("timestamp")

        df_features = build_feature_matrix(df_indexed, drop_na=True)

        if df_features.empty:
            return None

        detector = AnomalyDetector()
        scored = detector.score_dataframe(df_features)

        return scored

    except Exception as e:
        st.warning(f"Anomaly scoring unavailable: {e}")
        return None


# ── Sidebar ──


def render_sidebar():
    """Render the sidebar with controls and model info."""
    st.sidebar.title("⚡ Smart-Lite Insight")
    st.sidebar.markdown("---")

    # Site selector
    site_id = st.sidebar.selectbox("Site", ["home-01", "home-synth"], index=0)

    # Date range
    min_date, max_date = get_date_range(site_id)

    st.sidebar.subheader("Date Range")

    # Quick select buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Last 24h", use_container_width=True):
            st.session_state["start_date"] = (max_date - timedelta(hours=24)).date()
            st.session_state["end_date"] = max_date.date()
    with col2:
        if st.button("Last 7d", use_container_width=True):
            st.session_state["start_date"] = (max_date - timedelta(days=7)).date()
            st.session_state["end_date"] = max_date.date()

    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("Last 30d", use_container_width=True):
            st.session_state["start_date"] = (max_date - timedelta(days=30)).date()
            st.session_state["end_date"] = max_date.date()
    with col4:
        if st.button("All Data", use_container_width=True):
            st.session_state["start_date"] = min_date.date()
            st.session_state["end_date"] = max_date.date()

    # Default to last 7 days
    default_start = st.session_state.get(
        "start_date", (max_date - timedelta(days=7)).date()
    )
    default_end = st.session_state.get("end_date", max_date.date())

    start_date = st.sidebar.date_input(
        "Start",
        value=default_start,
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    end_date = st.sidebar.date_input(
        "End", value=default_end, min_value=min_date.date(), max_value=max_date.date()
    )

    # Resample interval
    resample = st.sidebar.selectbox(
        "Resample",
        ["None (1-min raw)", "5 min", "15 min", "1 hour", "1 day"],
        index=2,
    )
    resample_map = {
        "None (1-min raw)": None,
        "5 min": "5min",
        "15 min": "15min",
        "1 hour": "1h",
        "1 day": "1D",
    }

    # Anomaly detection toggle
    show_anomalies = st.sidebar.checkbox("Show Anomalies", value=True)

    st.sidebar.markdown("---")

    # Model info
    registry = load_model_info()
    if registry and registry.get("models"):
        latest = registry["models"][-1]
        st.sidebar.subheader("Model Info")
        st.sidebar.markdown(f"""
        **{latest['model_name'].replace('_', ' ').title()}** v{latest['version']}

        Trained: {latest['training_date'][:10]}
        Samples: {latest['n_training_samples']:,}
        Anomaly rate: {latest['anomaly_rate']:.2%}
        Features: {len(latest['feature_names'])}
        """)
    else:
        st.sidebar.warning("No trained model found")

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit + Plotly")

    return {
        "site_id": site_id,
        "start_date": start_date,
        "end_date": end_date,
        "resample": resample_map[resample],
        "show_anomalies": show_anomalies,
    }


# ── Main Charts ──


def render_consumption_chart(
    df: pd.DataFrame, scored: pd.DataFrame | None, resample: str | None
):
    """Render the main consumption chart with optional anomaly overlay."""
    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    plot_df = plot_df.set_index("timestamp")

    if resample:
        plot_df = plot_df.resample(resample).mean()

    fig = go.Figure()

    # Main consumption line
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["global_active_power_kw"],
            mode="lines",
            name="Active Power (kW)",
            line=dict(color="#4e79a7", width=1),
            opacity=0.8,
        )
    )

    # Anomaly overlay
    if scored is not None and not scored.empty:
        anomalies = scored[scored["is_anomaly"]]

        if resample:
            # For resampled data, mark the nearest resampled point
            anomaly_times = anomalies.index
            nearest = plot_df.index.get_indexer(anomaly_times, method="nearest")
            anomaly_points = plot_df.iloc[np.unique(nearest)]
        else:
            # For raw data, overlay directly
            anomaly_points = anomalies

        if not anomaly_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points.index,
                    y=anomaly_points["global_active_power_kw"],
                    mode="markers",
                    name=f"Anomalies ({len(anomaly_points):,})",
                    marker=dict(
                        color="#e15759",
                        size=6,
                        symbol="circle",
                        line=dict(width=1, color="white"),
                    ),
                )
            )

    fig.update_layout(
        title="Energy Consumption",
        xaxis_title="Time",
        yaxis_title="Active Power (kW)",
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_voltage_chart(df: pd.DataFrame, resample: str | None):
    """Render voltage stability chart."""
    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    plot_df = plot_df.set_index("timestamp")

    if resample:
        plot_df = plot_df.resample(resample).agg({"voltage_v": ["mean", "min", "max"]})
        plot_df.columns = ["mean", "min", "max"]
    else:
        plot_df = plot_df[["voltage_v"]].rename(columns={"voltage_v": "mean"})

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["mean"],
            mode="lines",
            name="Voltage (V)",
            line=dict(color="#59a14f", width=1),
        )
    )

    # EU nominal voltage reference lines
    fig.add_hline(
        y=230,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Nominal (230V)",
    )
    fig.add_hline(
        y=253, line_dash="dot", line_color="red", opacity=0.3, annotation_text="+10%"
    )
    fig.add_hline(
        y=207, line_dash="dot", line_color="red", opacity=0.3, annotation_text="-10%"
    )

    fig.update_layout(
        title="Voltage Stability",
        xaxis_title="Time",
        yaxis_title="Voltage (V)",
        height=300,
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_submetering_chart(df: pd.DataFrame, resample: str | None):
    """Render sub-metering breakdown over time."""
    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    plot_df = plot_df.set_index("timestamp")

    if resample:
        plot_df = plot_df.resample(resample).mean()

    fig = go.Figure()

    colors = {
        "sub_metering_1_wh": "#4e79a7",
        "sub_metering_2_wh": "#f28e2b",
        "sub_metering_3_wh": "#e15759",
    }
    names = {
        "sub_metering_1_wh": "Kitchen",
        "sub_metering_2_wh": "Laundry",
        "sub_metering_3_wh": "Water/AC",
    }

    for col, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[col],
                mode="lines",
                name=names[col],
                line=dict(color=color, width=1),
                stackgroup="one",
            )
        )

    fig.update_layout(
        title="Sub-Metering Breakdown",
        xaxis_title="Time",
        yaxis_title="Energy (Wh)",
        height=350,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


# ── Summary Metrics ──


def render_metrics(df: pd.DataFrame, scored: pd.DataFrame | None):
    """Render summary metric cards at the top."""
    col1, col2, col3, col4, col5 = st.columns(5)

    power = df["global_active_power_kw"]
    with col1:
        st.metric("Avg Power", f"{power.mean():.2f} kW")
    with col2:
        st.metric("Peak Power", f"{power.max():.2f} kW")
    with col3:
        st.metric("Readings", f"{len(df):,}")
    with col4:
        avg_voltage = df["voltage_v"].mean()
        st.metric("Avg Voltage", f"{avg_voltage:.1f} V")
    with col5:
        if scored is not None and not scored.empty:
            n_anomalies = scored["is_anomaly"].sum()
            rate = n_anomalies / len(scored) * 100
            st.metric("Anomalies", f"{n_anomalies:,}", f"{rate:.1f}%")
        else:
            st.metric("Anomalies", "N/A")


def render_anomaly_table(scored: pd.DataFrame | None):
    """Render a table of detected anomalies."""
    if scored is None or scored.empty:
        st.info("No anomaly data available. Ensure a model is trained.")
        return

    anomalies = scored[scored["is_anomaly"]].sort_values("anomaly_score")

    if anomalies.empty:
        st.success("No anomalies detected in this time range.")
        return

    st.subheader(f"Detected Anomalies ({len(anomalies):,})")

    display_cols = [
        "global_active_power_kw",
        "voltage_v",
        "anomaly_score",
    ]
    display_df = anomalies[display_cols].copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    display_df.columns = ["Power (kW)", "Voltage (V)", "Anomaly Score"]

    display_df["Power (kW)"] = display_df["Power (kW)"].round(3)
    display_df["Voltage (V)"] = display_df["Voltage (V)"].round(1)
    display_df["Anomaly Score"] = display_df["Anomaly Score"].round(4)

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        height=400,
    )


# ── Main App ──


def main():
    """Main application entry point."""
    params = render_sidebar()

    # Tabs
    tab_dashboard, tab_chat = st.tabs(["📊 Dashboard", "💬 AI Chat"])

    with tab_dashboard:
        # Load data
        start_str = params["start_date"].isoformat()
        end_str = (params["end_date"] + timedelta(days=1)).isoformat()

        df = load_data(params["site_id"], start_str, end_str)

        if df.empty:
            st.warning("No data found for the selected time range and site.")
        else:
            # Score for anomalies if requested
            scored = None
            if params["show_anomalies"]:
                with st.spinner("Scoring for anomalies..."):
                    scored = score_data(df)

            # Render
            render_metrics(df, scored)
            st.markdown("---")

            render_consumption_chart(df, scored, params["resample"])

            col_left, col_right = st.columns(2)
            with col_left:
                render_voltage_chart(df, params["resample"])
            with col_right:
                render_submetering_chart(df, params["resample"])

            st.markdown("---")

            if params["show_anomalies"]:
                render_anomaly_table(scored)

    with tab_chat:
        try:
            from dashboard.chat import render_chat_tab
        except ModuleNotFoundError:
            from chat import render_chat_tab

        render_chat_tab()


if __name__ == "__main__":
    main()
