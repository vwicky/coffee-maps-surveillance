"""
Streamlit analytics dashboard: reads face/session data from MongoDB (same schema as MongoMetadataManager).

Run from project root:
  pip install streamlit plotly
  streamlit run dashboard.py
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "coffee_shop_analytics")


def _connect() -> MongoClient:
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI is not set. Add it to .env or the environment.")
    return MongoClient(MONGO_URI)


@st.cache_data(ttl=60)
def load_faces_dataframe() -> pd.DataFrame:
    client = _connect()
    try:
        coll = client[MONGO_DB]["sessions"]
        sessions: list[dict[str, Any]] = list(
            coll.find({}, {"session_id": 1, "faces": 1})
        )
    finally:
        client.close()

    rows: list[dict[str, Any]] = []
    for doc in sessions:
        sid = doc.get("session_id") or ""
        for face in doc.get("faces") or []:
            cr = face.get("classification_result") or {}
            am = face.get("additional_metadata") or {}
            age = cr.get("age")
            try:
                age_i = int(age) if age is not None else 0
            except (TypeError, ValueError):
                age_i = 0
            sex = str(cr.get("sex") or "").strip().upper()
            dur = float(face.get("total_duration_sec") or am.get("duration_sec") or 0.0)
            rows.append(
                {
                    "session_id": sid,
                    "age": age_i,
                    "sex": sex,
                    "duration_sec": dur,
                }
            )

    return pd.DataFrame(rows)


def sex_label(s: str) -> str:
    if s == "M":
        return "Male"
    if s == "F":
        return "Female"
    return "Unknown / other"


def age_group(age: int) -> str:
    if age is None or age <= 0:
        return "Unknown"
    if age < 18:
        return "0–17"
    if age <= 25:
        return "18–25"
    if age <= 35:
        return "26–35"
    if age <= 45:
        return "36–45"
    if age <= 55:
        return "46–55"
    return "56+"


def main() -> None:
    st.set_page_config(page_title="Coffee shop analytics", layout="wide")
    st.title("Visitor analytics (MongoDB)")

    try:
        df = load_faces_dataframe()
    except Exception as e:
        st.error(f"Could not load data from MongoDB: {e}")
        st.stop()

    if df.empty:
        st.warning("No face documents found. Run the pipeline so sessions are written to MongoDB.")
        st.stop()

    df = df[df['duration_sec'] > 0.0]

    sessions = ["All sessions"] + sorted(df["session_id"].dropna().unique().tolist())
    choice = st.sidebar.selectbox("Session filter", sessions)
    if choice != "All sessions":
        df = df[df["session_id"] == choice]

    df["sex_display"] = df["sex"].apply(sex_label)
    df["age_bucket"] = df["age"].apply(age_group)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sex distribution")
        sex_counts = df["sex_display"].value_counts().reset_index()
        sex_counts.columns = ["sex", "count"]
        fig_pie = px.pie(
            sex_counts,
            names="sex",
            values="count",
            hole=0.35,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Age groups")
        order = ["0–17", "18–25", "26–35", "36–45", "46–55", "56+", "Unknown"]
        vc = df["age_bucket"].value_counts()
        age_counts = pd.DataFrame(
            {"age_group": order, "count": [int(vc.get(g, 0)) for g in order]}
        )
        fig_bar = px.bar(
            age_counts,
            x="age_group",
            y="count",
            category_orders={"age_group": order},
        )
        fig_bar.update_layout(xaxis_title="Age group", yaxis_title="People")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Time spent in establishment")
    avg_min = df["duration_sec"].mean() / 60.0 if len(df) else 0.0
    st.metric("Average visit duration (faces in selection)", f"{avg_min:.2f} min")

    # Per-session average (useful when "All sessions" is selected)
    per_session = (
        df.groupby("session_id", as_index=False)["duration_sec"]
        .mean()
        .rename(columns={"duration_sec": "avg_duration_sec"})
      )
    per_session["avg_minutes"] = per_session["avg_duration_sec"] / 60.0
    per_session = per_session.sort_values("session_id")

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=per_session["session_id"],
            y=per_session["avg_minutes"],
            mode="lines+markers",
            name="Avg minutes / session",
        )
    )
    fig_line.update_layout(
        xaxis_title="Session",
        yaxis_title="Average duration (minutes)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("Raw rows (sample)"):
        st.dataframe(df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
