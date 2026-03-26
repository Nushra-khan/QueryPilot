"""
agent.py — QueryPilot
----------------------
Creates and runs the ReAct Research Agent.
QueryPilot uses the Thought → Action → Observation loop
to autonomously gather research before writing the final report.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from tools import get_all_tools


# ─────────────────────────────────────────────
# ReAct Prompt
# ─────────────────────────────────────────────

REACT_PROMPT = """You are QueryPilot — an expert autonomous research agent. 
Your mission is to thoroughly research the given topic and gather comprehensive, 
accurate information from multiple sources before writing your report.

You have access to the following tools:

{tools}

RESEARCH STRATEGY:
1. Start with a broad web search to find recent and relevant information
2. Use Wikipedia for foundational background knowledge
3. Perform 2-3 more focused web searches on subtopics (findings, challenges, future trends)
4. Only generate your Final Answer after at least 4 tool uses

Use this EXACT format:

Thought: [Reason about what information you need next]
Action: [one of: {tool_names}]
Action Input: [your search query]
Observation: [result from the tool]
... (repeat minimum 4 times)

Thought: I now have comprehensive information to write a full report.
Final Answer: [Compile ALL gathered information with these sections:
  INTRODUCTION: [topic overview — 2 paragraphs]
  KEY FINDINGS: [5+ numbered findings with full details]
  CHALLENGES: [3+ obstacles or issues]
  FUTURE SCOPE: [3+ upcoming trends]
  CONCLUSION: [2 paragraph summary]
]

RULES:
- Use tools AT LEAST 4 times before Final Answer
- Never fabricate statistics — use only what tools return
- Be thorough; the Final Answer is the raw material for the full report

Begin!

Topic: {input}

{agent_scratchpad}"""


# ─────────────────────────────────────────────
# Progress Callback
# ─────────────────────────────────────────────

class QueryPilotCallback(BaseCallbackHandler):
    """Streams live ReAct steps to the UI."""

    def __init__(self, step_callback=None):
        self.steps = []
        self.step_callback = step_callback

    def _emit(self, msg: str):
        self.steps.append(msg)
        if self.step_callback:
            self.step_callback(msg)

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        preview = input_str[:80] + ("..." if len(input_str) > 80 else "")
        self._emit(f"🔍 **{name}** → `{preview}`")

    def on_tool_end(self, output, **kwargs):
        preview = str(output)[:120].replace("\n", " ")
        self._emit(f"📄 Result: `{preview}{'...' if len(str(output)) > 120 else ''}`")

    def on_agent_action(self, action, **kwargs):
        self._emit(f"💭 Choosing tool: `{action.tool}`")

    def on_agent_finish(self, finish, **kwargs):
        self._emit("✅ Research complete — formatting report...")


# ─────────────────────────────────────────────
# Agent Factory
# ─────────────────────────────────────────────

def create_querypilot_agent(step_callback=None):
    """Builds the QueryPilot ReAct agent executor."""

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=8192,
        convert_system_message_to_human=True,
    )

    tools = get_all_tools()

    prompt = PromptTemplate(
        template=REACT_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"- {t.name}: {t.description}" for t in tools]),
            "tool_names": ", ".join([t.name for t in tools]),
        },
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    callback = QueryPilotCallback(step_callback=step_callback)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        max_execution_time=180,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        callbacks=[callback],
        return_intermediate_steps=True,
    )

    return executor, callback


# ─────────────────────────────────────────────
# Run Agent
# ─────────────────────────────────────────────

def run_querypilot(topic: str, step_callback=None) -> dict[str, Any]:
    """
    Runs QueryPilot on the given topic.
    Returns dict: output, steps, intermediate
    """
    executor, callback = create_querypilot_agent(step_callback=step_callback)
    result = executor.invoke({"input": topic})

    return {
        "output": result.get("output", ""),
        "steps": callback.steps,
        "intermediate": result.get("intermediate_steps", []),
    }


# ─────────────────────────────────────────────
# LLM Factory (used by report_generator)
# ─────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=api_key,
        temperature=0.4,
        max_output_tokens=8192,
        convert_system_message_to_human=True,
    )