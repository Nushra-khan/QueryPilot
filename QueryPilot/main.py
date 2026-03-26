"""
main.py — QueryPilot
---------------------
Backend pipeline: Agent → Raw Research → Report Formatter → Final Report

Can be run standalone (CLI) or imported by app.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from agent import run_querypilot, get_llm
from report_generator import format_report, report_to_plain_text


def generate_research_report(
    topic: str,
    step_callback=None,
    save_to_file: bool = False,
) -> dict:
    """
    Full QueryPilot pipeline: Research → Format → Return
    """

    start = time.time()

    if step_callback:
        step_callback("🧭 QueryPilot initializing research agent...")

    print(f"\n{'='*60}")
    print(f"  🧭 QUERYPILOT — Autonomous Research Agent")
    print(f"  Topic   : {topic}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── 1. Run Agent ──
    agent_result = run_querypilot(topic=topic, step_callback=step_callback)
    raw_output   = agent_result["output"]
    steps        = agent_result["steps"]
    intermediate = agent_result["intermediate"]

    # ── 2. Build context ──
    parts = [f"TOPIC: {topic}\n"]

    for action, observation in intermediate:
        if hasattr(action, "tool") and hasattr(action, "tool_input"):
            parts.append(
                f"\n[Tool: {action.tool} | Query: {action.tool_input}]\n{observation}\n"
            )

    parts.append(f"\nAGENT SUMMARY:\n{raw_output}")
    combined = "\n".join(parts)

    # ── 3. Format report ──
    if step_callback:
        step_callback("📝 QueryPilot is formatting your report...")

    llm = get_llm()
    report_md = format_report(topic=topic, raw_research=combined, llm=llm)

    # ── 4. Save (optional) ──
    file_path = _save_report(topic, report_md) if save_to_file else None

    duration = round(time.time() - start, 1)
    print(f"\n✅ QueryPilot finished in {duration}s\n{'='*60}\n")

    return {
        "report_md": report_md,
        "raw_output": raw_output,
        "steps": steps,
        "topic": topic,
        "duration": duration,
        "file_path": file_path,
    }


def _save_report(topic: str, report_md: str) -> str:
    """Save report to file"""

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)
    safe = safe.strip().replace(" ", "_")[:60]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"QueryPilot_{safe}_{timestamp}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("QUERYPILOT RESEARCH REPORT\n")
        f.write(f"Topic     : {topic}\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_md)

    print(f"[QueryPilot] Report saved → {path}")
    return str(path)


# ── CLI ENTRY ──

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage  : python main.py \"Your Research Topic\"")
        sys.exit(1)

    import re

    topic = " ".join(sys.argv[1:])

    def cli_cb(msg):
        print(" ", re.sub(r"[*`#]", "", msg))

    result = generate_research_report(
        topic=topic,
        step_callback=cli_cb,
        save_to_file=True
    )

    print("\n" + "=" * 60)
    print(result["report_md"][:2000])
    print("\n... [see full report in reports/ folder]")
    print(f"\n✅ Saved → {result['file_path']}")
    print(f"⏱️ Time → {result['duration']}s")