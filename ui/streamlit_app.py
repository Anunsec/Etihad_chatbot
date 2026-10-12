"""Streamlit front end for guest chat, saved conversations, and admin metrics."""

from __future__ import annotations

import base64
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
LOGO_PATH = Path(__file__).resolve().parents[1] / "OIP.webp"


# Static theme and brand header are rendered once per Streamlit script run.
st.set_page_config(
    page_title="Travel Assistant | Etihad",
    page_icon=str(LOGO_PATH),
    layout="wide",
)
st.markdown(
    f"<style>{Path(__file__).with_name('theme.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
st.markdown(
    f"""
    <div class="brand-bar">
      <img src="data:image/webp;base64,{logo_data}" alt="Etihad Airways logo">
      <div>
        <div class="brand-topline">Etihad Airways inspired</div>
        <div class="brand-product">Travel assistant</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Send a backend request with the current login token when present."""
    headers = kwargs.pop("headers", {})
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=60, **kwargs)


def submit_feedback(answer_id: str, rating: int) -> None:
    """Persist one answer rating and show a small status message."""
    try:
        response = api_request(
            "POST", "/feedback", json={"answer_id": answer_id, "rating": rating}
        )
        response.raise_for_status()
        st.toast("Feedback saved")
    except httpx.HTTPError:
        st.error("Could not save feedback. Please try again.")


def format_usd(value: float) -> str:
    """Retain enough decimals to show small per-chat cost estimates."""
    amount = Decimal(str(value))
    precision = Decimal("0.00000001") if 0 < amount < Decimal("0.01") else Decimal("0.01")
    places = 8 if precision < Decimal("0.01") else 2
    return f"${amount.quantize(precision, rounding=ROUND_HALF_UP):,.{places}f}"


def reset_session() -> None:
    """Remove login, guest profile, and chat state when signing out."""
    for key in (
        "token", "role", "messages", "conversation_id", "active_view",
        "open_assistant_view", "guest_name", "guest_email",
        "guest_name_input", "guest_email_input",
    ):
        st.session_state.pop(key, None)


if "messages" not in st.session_state:
    # Streamlit reruns the script; session_state is the UI's cross-rerun memory.
    st.session_state.messages = []

if not st.session_state.get("token"):
    # The backend owns authentication; the UI only keeps its bearer token.
    welcome, access = st.columns([1.25, 1], gap="large", vertical_alignment="center")
    with welcome:
        st.markdown(
            """
            <div class="hero">
              <div class="eyebrow">WELCOME ABOARD</div>
              <h1>A smoother journey starts here.</h1>
              <p>Ask about travel policies, check a flight, or find schedule information
              in one simple conversation.</p>
            </div>
            <p class="login-copy">Policy guidance is linked to its source. Flight answers use
            AviationStack data when available.</p>
            """,
            unsafe_allow_html=True,
        )
    with access:
        st.markdown('<div class="eyebrow-dark">YOUR ACCOUNT</div>', unsafe_allow_html=True)
        st.subheader("Sign in to continue")
        with st.form("login"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
    if submitted:
        try:
            response = httpx.post(
                f"{API_URL}/auth/token",
                data={"username": username, "password": password},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            st.session_state.token = payload["access_token"]
            st.session_state.role = payload["role"]
            st.rerun()
        except Exception as exc:
            st.error(f"Sign-in failed: {exc}")
    st.stop()

with st.sidebar:
    # Saved chats come from SQLite via the authenticated API, not browser storage.
    st.markdown('<div class="eyebrow">YOUR JOURNEY</div>', unsafe_allow_html=True)
    st.markdown("### Travel assistant")
    st.caption(f"Signed in as {st.session_state.role.title()}")
    if st.session_state.get("guest_name"):
        st.caption(f"Welcome, {st.session_state.guest_name}")
    st.divider()
    if st.session_state.pop("open_assistant_view", False):
        st.session_state.active_view = "Assistant"
    page = (
        st.radio(
            "View", ["Assistant", "Admin dashboard"],
            label_visibility="collapsed", key="active_view"
        )
        if st.session_state.role == "admin"
        else "Assistant"
    )
    st.divider()
    if st.button("＋  New conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("conversation_id", None)
        st.session_state.open_assistant_view = True
        st.rerun()
    st.markdown("#### Recent conversations")
    try:
        history_response = api_request("GET", "/conversations?limit=50")
        history_response.raise_for_status()
        conversations = history_response.json()
    except httpx.HTTPError:
        conversations = []
        st.caption("Could not load saved conversations right now.")
    if not conversations:
        st.caption("Your saved chats will appear here.")
    for conversation in conversations:
        label = conversation["title"] or "Conversation"
        if st.button(label, key=f"conversation-{conversation['id']}", width="stretch"):
            try:
                detail_response = api_request("GET", f"/conversations/{conversation['id']}")
                detail_response.raise_for_status()
                detail = detail_response.json()
                st.session_state.conversation_id = detail["conversation_id"]
                st.session_state.messages = detail["messages"]
                st.session_state.open_assistant_view = True
                st.rerun()
            except httpx.HTTPError:
                st.error("Could not open this conversation. Please try again.")
    if st.button("Sign out", width="stretch"):
        reset_session()
        st.rerun()



if page == "Admin dashboard":
    # The backend enforces admin RBAC; hiding this view is only a UI convenience.
    st.markdown(
        """
        <div class="hero admin-hero">
          <div class="eyebrow">ADMIN OVERVIEW</div>
          <h1>A clear view of every answer.</h1>
          <p>Track assistant activity, model use, policy retrieval, and cache performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if (
        st.button("Refresh data", type="primary")
        or "admin_dashboard" not in st.session_state
        or "cost" not in st.session_state.admin_dashboard
    ):
        # Keep dashboard data in the session until the admin explicitly refreshes.
        try:
            dashboard_response = api_request("GET", "/admin/dashboard?limit=250")
            dashboard_response.raise_for_status()
            st.session_state.admin_dashboard = dashboard_response.json()
            audit_response = api_request("GET", "/admin/audit-logs?limit=250")
            audit_response.raise_for_status()
            st.session_state.audit_logs = audit_response.json()
        except Exception as exc:
            st.error(f"Could not load dashboard: {exc}")

    dashboard = st.session_state.get("admin_dashboard")
    if dashboard and "cost" not in dashboard:
        st.warning("Restart the FastAPI backend, then refresh this dashboard to see cost data.")
        st.stop()
    if dashboard:
        config = dashboard["configuration"]
        st.caption(
            f"OpenAI: {'configured' if config['openai_configured'] else 'missing'} · "
            f"Model: {config['openai_model']} · "
            f"LangSmith: {'enabled' if config['langsmith_tracing'] else 'disabled'} · "
            f"AviationStack: {'configured' if config['aviationstack_configured'] else 'missing'}"
        )
        llm = dashboard["llm"]
        cost = dashboard["cost"]
        cost_display = (
            format_usd(cost["estimated_total_usd"])
            if cost["priced_calls"] or not llm["total_calls"] else "Unpriced"
        )
        app = dashboard["application"]
        audit_logs = st.session_state.get("audit_logs", [])
        overview_tab, usage_tab, activity_tab = st.tabs(
            ["Overview", "Model usage", "Activity & logs"]
        )

        with overview_tab:
            st.markdown('<div class="section-intro">At a glance</div>', unsafe_allow_html=True)
            metric_columns = st.columns(5)
            metric_columns[0].metric("LLM calls", llm["total_calls"])
            metric_columns[1].metric("Successful", llm["successful_calls"])
            metric_columns[2].metric(
                "Estimated chat cost", cost_display
            )
            metric_columns[3].metric("Cached answers", app["policy_cache_entries"])
            metric_columns[4].metric("Cache hits", app["policy_cache_hits"])

            st.markdown('<div class="section-intro">Application activity</div>', unsafe_allow_html=True)
            app_columns = st.columns(4)
            app_columns[0].metric("Users", app["users"])
            app_columns[1].metric("Conversations", app["conversations"])
            app_columns[2].metric("Messages", app["messages"])
            app_columns[3].metric("Feedback", app["feedback_total"])
            st.caption(
                f"Cache: {app.get('policy_cache_backend', 'unavailable')} · "
                f"RAG: {app.get('rag_backend', 'local_bm25')} · "
                f"New policy chunks indexed: {app.get('rag_indexed_new_chunks', 0)}"
            )
            for warning in app.get("storage_warnings", []):
                st.warning(warning)

        with usage_tab:
            st.markdown('<div class="section-intro">Model usage</div>', unsafe_allow_html=True)
            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Estimated chat cost", cost_display
            )
            metric_columns[1].metric("Total tokens", llm["total_tokens"])
            metric_columns[2].metric("Failed calls", llm["failed_calls"])
            metric_columns[3].metric("Average latency", f"{llm['average_latency_ms']:.0f} ms")
            st.caption(
                f"USD estimate for {cost['priced_calls']} recorded OpenAI chat calls using "
                f"[published standard rates]({cost['pricing_source']}) as of "
                f"{cost['pricing_as_of']}. Excludes embeddings, AviationStack, taxes, and discounts."
            )
            if cost["unpriced_calls"]:
                st.warning(
                    f"{cost['unpriced_calls']} call(s) could not be priced because the model "
                    "or token usage was unavailable; the estimate is incomplete."
                )
            token_columns = st.columns(4)
            token_columns[0].metric("Input", llm["input_tokens"])
            token_columns[1].metric("Output", llm["output_tokens"])
            token_columns[2].metric("Cached input", llm["cached_input_tokens"])
            token_columns[3].metric("Reasoning", llm["reasoning_tokens"])
            if dashboard["daily"]:
                chart_left, chart_right = st.columns(2)
                with chart_left:
                    st.subheader("Calls over time")
                    st.bar_chart(dashboard["daily"], x="day", y="calls")
                with chart_right:
                    st.subheader("Tokens over time")
                    st.bar_chart(dashboard["daily"], x="day", y="total_tokens")
                st.subheader("Estimated chat cost over time (USD)")
                st.bar_chart(dashboard["daily"], x="day", y="estimated_cost_usd")
            else:
                st.info("No LLM calls have been recorded yet.")
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.subheader("By model")
                st.dataframe(dashboard["by_model"], width="stretch", hide_index=True)
            with chart_right:
                st.subheader("By graph node")
                st.dataframe(dashboard["by_agent"], width="stretch", hide_index=True)

        with activity_tab:
            st.markdown('<div class="section-intro">Answer outcomes</div>', unsafe_allow_html=True)
            st.dataframe(dashboard["decisions"], width="stretch", hide_index=True)
            st.markdown('<div class="section-intro">Recent LLM calls</div>', unsafe_allow_html=True)
            safe_calls = [
                {key: value for key, value in row.items() if key not in {"user_id", "conversation_id"}}
                for row in dashboard["recent_calls"]
            ]
            st.dataframe(safe_calls, width="stretch", hide_index=True)
            st.markdown('<div class="section-intro">Recent audit events</div>', unsafe_allow_html=True)
            st.dataframe(audit_logs, width="stretch", hide_index=True)
            download_left, download_right = st.columns(2)
            download_left.download_button(
                "Download LLM usage JSON",
                data=json.dumps(dashboard, indent=2),
                file_name="llm-usage-dashboard.json",
                mime="application/json",
            )
            download_right.download_button(
                "Download audit JSON",
                data=json.dumps(audit_logs, indent=2),
                file_name="audit-log.json",
                mime="application/json",
            )
    st.stop()


if not st.session_state.get("guest_name") or not st.session_state.get("guest_email"):
    # Guest name/email personalize this UI session; they are not sent in chat prompts.
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">WELCOME ABOARD</div>
          <h1>How can we assist you?</h1>
          <p>Before we begin, tell us how to address you during this visit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("guest_welcome"):
        st.subheader("A quick introduction")
        name = st.text_input(
            "Your name", max_chars=80, placeholder="What should we call you?",
            key="guest_name_input",
        )
        email = st.text_input(
            "Email address", type="email", max_chars=254,
            key="guest_email_input",
        )
        submitted = st.form_submit_button(
            "Start chatting", type="primary", width="stretch", key="guest_start"
        )
    st.caption("These details are used only for this chat session and are not added to answers.")
    if submitted:
        clean_name = " ".join(name.split())
        clean_email = email.strip().lower()
        if not clean_name:
            st.error("Please enter your name.")
        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_email):
            st.error("Please enter a valid email address.")
        else:
            st.session_state.guest_name = clean_name
            st.session_state.guest_email = clean_email
            st.rerun()
    st.stop()


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">HOW CAN WE HELP?</div>
      <h1>Your journey, made simple.</h1>
      <p>Find the policy detail you need or ask for current flight information.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.fragment
def chat_panel() -> None:
    """Render chat independently so submitting a prompt avoids a full-page rerun."""
    with st.chat_message("assistant"):
        st.write(f"Hello, {st.session_state.guest_name}! How can we assist you today?")
    starter_prompt = None
    if not st.session_state.messages:
        st.markdown('<div class="section-intro">Explore a topic</div>', unsafe_allow_html=True)
        policy_col, flight_col, arrivals_col = st.columns(3)
        if policy_col.button("Economy baggage allowance", width="stretch"):
            starter_prompt = "What is the baggage allowance for Economy Value?"
        if flight_col.button("Check a flight status", width="stretch"):
            starter_prompt = "What is the status of flight EY 219 today?"
        if arrivals_col.button("Arrivals at Delhi", width="stretch"):
            starter_prompt = "Show flights arriving at Delhi airport."
        st.markdown(
            '<p class="quiet-note">Answers include source details when available. '
            'Flight information can change.</p>',
            unsafe_allow_html=True,
        )

    for index, item in enumerate(st.session_state.messages):
        # Replayed messages include source and feedback controls for saved answers.
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                decision_labels = {
                    "clarify": "One more detail needed",
                    "not_found": "No verified match found",
                    "tool_unavailable": "Information temporarily unavailable",
                }
                if item.get("decision") in decision_labels:
                    st.caption(decision_labels[item["decision"]])
                citations = item.get("citations", [])
                if citations:
                    with st.expander(f"View {len(citations)} source{'s' if len(citations) != 1 else ''}"):
                        for citation_index, citation in enumerate(citations):
                            if citation_index:
                                st.divider()
                            st.markdown(f"**{citation['document_title']}**")
                            st.caption(citation["section_path"])
                            if citation.get("canonical_url"):
                                st.markdown(f"[Open source ↗]({citation['canonical_url']})")
                            if citation.get("snapshot_page"):
                                st.caption(f"Snapshot page {citation['snapshot_page']}")
                            st.markdown(citation["passage"])
                            if citation.get("table_cell_ids"):
                                st.caption("Cells: " + ", ".join(citation["table_cell_ids"]))
                col_up, col_down, _ = st.columns([1.2, 1.2, 5])
                if col_up.button("Helpful", key=f"up-{index}", width="stretch"):
                    submit_feedback(item["answer_id"], 1)
                if col_down.button("Needs work", key=f"down-{index}", width="stretch"):
                    submit_feedback(item["answer_id"], -1)

    entered_prompt = st.bottom.chat_input(
        "Ask about baggage, policies, flights, or arrivals..."
    )
    if prompt := starter_prompt or entered_prompt:
        # Show the user's turn immediately, then append the backend response.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response = api_request(
                    "POST",
                    "/chat",
                    json={
                        "message": prompt,
                        "conversation_id": st.session_state.get("conversation_id"),
                    },
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.conversation_id = result["conversation_id"]
                st.markdown(result["answer_markdown"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer_markdown"],
                        "decision": result["decision"],
                        "citations": result["citations"],
                        "answer_id": result["answer_id"],
                    }
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    reset_session()
                    st.error("Your session expired. Please sign in again.")
                else:
                    st.error(exc.response.text)
            except Exception as exc:
                st.error(f"Backend unavailable: {exc}")


chat_panel()
