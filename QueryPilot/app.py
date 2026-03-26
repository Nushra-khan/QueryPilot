"""
app.py — QueryPilot
--------------------
Streamlit UI for QueryPilot: Autonomous Research Agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from main import generate_research_report, report_to_plain_text
from langchain.agents import AgentExecutor, create_react_agent

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="QueryPilot - Your Autonomous Research Agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Main header ── */
  .qp-header {
    text-align: center;
    padding: 2rem 0 0.8rem 0;
  }
  .qp-logo {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #3ECFCF, #6C63FF);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
  }
  .qp-tagline {
    color: #888;
    font-size: 1.05rem;
    margin-top: -4px;
  }
  .qp-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6C63FF22, #3ECFCF22);
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    color: #6C63FF;
    font-weight: 600;
    margin-top: 6px;
  }

  /* ── Input area ── */
  .stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #e0e0e0 !important;
    padding: 0.8rem 1rem !important;
    font-size: 1.05rem !important;
    transition: border-color 0.2s;
  }
  .stTextInput > div > div > input:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 3px #6C63FF22 !important;
  }

  /* ── Generate button ── */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #3ECFCF) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    color: white !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.35) !important;
  }

  /* ── Sample topic buttons ── */
  div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    border: 1px solid #6C63FF55 !important;
    color: #6C63FF !important;
    background: transparent !important;
  }
  div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #6C63FF11 !important;
  }

  /* ── Download buttons ── */
  .stDownloadButton > button {
    border-radius: 10px !important;
    width: 100% !important;
    font-weight: 600 !important;
  }

  /* ── Cover card ── */
  .qp-cover {
    background: linear-gradient(135deg, #1a1035, #0d2137);
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .qp-cover h2 { color: #a78bfa; margin: 0; font-size: 1.5rem; }
  .qp-cover h3 { color: #ffffff; margin: 0.4rem 0; font-size: 1.25rem; }
  .qp-cover p  { color: #99a; margin: 0; font-size: 0.9rem; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #1a1035, #0d2137);
  }
  section[data-testid="stSidebar"] * { color: #ddd !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #a78bfa !important; }
  section[data-testid="stSidebar"] hr { border-color: #333 !important; }

  /* ── Metrics ── */
  div[data-testid="metric-container"] {
    background: #f8f7ff;
    border: 1px solid #e0deff;
    border-radius: 10px;
  }

  /* ── Empty state ── */
  .qp-empty {
    text-align: center;
    padding: 3.5rem 2rem;
    background: linear-gradient(135deg, #f8f7ff, #f0fffe);
    border-radius: 16px;
    margin-top: 2rem;
    border: 2px dashed #c4c1ff;
  }
  .qp-empty-icon { font-size: 4rem; }
  .qp-empty h3 { color: #6C63FF; margin: 0.5rem 0; }
  .qp-empty p  { color: #888; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

def init_session():
    defaults = {
        "report_md":  None,
        "steps":      [],
        "topic":      "",
        "duration":   0,
        "generating": False,
        "history":    [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("# 🧭 QueryPilot")
        st.markdown("*Your autonomous research companion*")
        st.markdown("---")

        st.markdown("### 🔄 How It Works")
        st.markdown("""
**1. Enter a Topic**
Type anything you want researched.

**2. Agent Takes Over**
QueryPilot autonomously:
- 🌐 Searches the web for recent data
- 📚 Queries Wikipedia for background
- 🔍 Runs multiple targeted searches
- 🧠 Reasons across all sources

**3. Report Delivered**
A structured report appears with:
- Introduction
- Key Findings
- Challenges
- Future Scope
- Conclusion
""")

        st.markdown("---")
        st.markdown("### 💡 Try These Topics")

        samples = [
            "Impact of AI in Healthcare",
            "Role of AI in Education",
            "Future of Electric Vehicles",
            "Quantum Computing Applications",
            "Blockchain in Supply Chain",
            "Climate Change & Renewable Energy",
            "Cybersecurity Trends 2025",
        ]
        for s in samples:
            if st.button(s, key=f"s_{s}", use_container_width=True):
                st.session_state["topic_input"] = s
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ API Status")
        gkey = os.getenv("GOOGLE_API_KEY", "")
        tkey = os.getenv("TAVILY_API_KEY", "")
        st.success("✅ Google Gemini") if gkey else st.error("❌ Google Gemini (required)")
        st.success("✅ Tavily Search") if tkey else st.warning("⚠️ Tavily (using DuckDuckGo)")

        if st.session_state["history"]:
            st.markdown("---")
            st.markdown("### 📁 Recent Reports")
            for i, (t, _) in enumerate(reversed(st.session_state["history"][-5:])):
                st.markdown(f"**{i+1}.** {t[:38]}{'…' if len(t) > 38 else ''}")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;font-size:0.78rem;color:#666;'>"
            "QueryPilot · LangChain + Gemini<br/>Built with Streamlit"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

def render_header():
    st.markdown("""
<div class="qp-header">
  <div class="qp-logo">🧭 QueryPilot</div>
  <div class="qp-tagline">Autonomous AI Research Agent — LangChain · Gemini · ReAct</div>
  <div class="qp-badge">⚡ Powered by Google Gemini 1.5</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")


# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────

def render_input():
    st.markdown("### 🔍 What do you want to research?")
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input(
            "topic",
            placeholder='e.g. "Impact of AI in Healthcare" or "Future of Quantum Computing"',
            value=st.session_state.get("topic_input", ""),
            key="topic_input",
            label_visibility="collapsed",
        )
    with col2:
        clicked = st.button(
            "🚀 Launch Research",
            type="primary",
            disabled=st.session_state["generating"],
            use_container_width=True,
        )
    return topic, clicked


# ─────────────────────────────────────────────
# Report Renderer
# ─────────────────────────────────────────────

def render_report(report_md, topic, duration, steps):
    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ Research Time",   f"{duration}s")
    c2.metric("🔍 Agent Steps",     len(steps))
    c3.metric("📝 Words Generated", f"~{len(report_md.split()):,}")
    c4.metric("📊 Status",          "✅ Complete")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Report", "🤖 Agent Steps", "⬇️ Download"])

    with tab1:
        today = datetime.now().strftime("%B %d, %Y")
        st.markdown(f"""
<div class="qp-cover">
  <h2>🧭 QueryPilot Research Report</h2>
  <h3>{topic.title()}</h3>
  <p>Generated by QueryPilot &nbsp;·&nbsp; Powered by Google Gemini &nbsp;·&nbsp; 📅 {today}</p>
</div>
""", unsafe_allow_html=True)
        st.markdown(report_md)

    with tab2:
        st.markdown("### 🤖 ReAct Reasoning Trace")
        st.caption("Every Thought → Action → Observation the agent performed:")
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")

    with tab3:
        st.markdown("### ⬇️ Download Your Report")
        plain = report_to_plain_text(report_md)
        safe  = topic.replace(" ", "_")[:40]
        today_s = datetime.now().strftime("%Y%m%d")

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "📄 Download .txt",
                data=plain,
                file_name=f"QueryPilot_{safe}_{today_s}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_b:
            st.download_button(
                "📝 Download .md",
                data=report_md,
                file_name=f"QueryPilot_{safe}_{today_s}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.info(
            "💡 **Save as PDF:** Press `Ctrl+P` / `Cmd+P` in your browser "
            "while on the Report tab → choose **Save as PDF**."
        )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()
    render_header()

    topic, clicked = render_input()

    if clicked and not topic.strip():
        st.warning("⚠️ Please enter a research topic first.")
        return

    if clicked and not os.getenv("GOOGLE_API_KEY", "").strip():
        st.error("""
❌ **GOOGLE_API_KEY not found!**

1. Create a `.env` file in the project root
2. Add: `GOOGLE_API_KEY=your_key_here`
3. Get your free key → [aistudio.google.com](https://aistudio.google.com)
""")
        return

    if clicked and topic.strip():
        st.session_state["generating"] = True
        st.session_state["report_md"]  = None

        live_steps        = []
        steps_placeholder = st.empty()

        def update(msg: str):
            live_steps.append(msg)
            with steps_placeholder.container():
                st.markdown("**🔄 QueryPilot is researching...**")
                for s in live_steps[-6:]:
                    st.markdown(f"> {s}")

        with st.spinner(f"🧭 QueryPilot is researching **{topic}** — this takes 1-3 minutes..."):
            try:
                result = generate_research_report(
                    topic=topic.strip(),
                    step_callback=update,
                    save_to_file=True,
                )
                st.session_state["report_md"] = result["report_md"]
                st.session_state["steps"]     = result["steps"]
                st.session_state["topic"]     = topic.strip()
                st.session_state["duration"]  = result["duration"]
                st.session_state["history"].append((topic.strip(), result["report_md"]))
                steps_placeholder.empty()
                st.success(f"✅ QueryPilot finished in {result['duration']}s!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback; print(traceback.format_exc())

        st.session_state["generating"] = False

    if st.session_state["report_md"]:
        render_report(
            st.session_state["report_md"],
            st.session_state["topic"],
            st.session_state["duration"],
            st.session_state["steps"],
        )
    elif not st.session_state["generating"]:
        st.markdown("""
<div class="qp-empty">
  <div class="qp-empty-icon">🧭</div>
  <h3>QueryPilot Ready for Takeoff</h3>
  <p>Enter any research topic above and hit <strong>Launch Research</strong>.<br/>
  QueryPilot will autonomously search, analyze, and generate a full report for you.</p>
  <br/>
  <p>💡 <em>Try: "Impact of AI in Healthcare" or "Future of Quantum Computing"</em></p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
