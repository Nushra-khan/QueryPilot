"""
tools.py — QueryPilot
----------------------
Tool definitions for the QueryPilot research agent.

Tools:
  1. web_search      → Tavily (preferred) or DuckDuckGo (free fallback)
  2. wikipedia_search → Wikipedia encyclopedic knowledge
"""

import os
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool


def get_web_search_tool() -> Tool:
    """
    Returns the best available web search tool.
    Uses Tavily if TAVILY_API_KEY is set, otherwise DuckDuckGo.
    """
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    if tavily_key:
        print("[QueryPilot] ✅ Web search: Tavily")
        search = TavilySearchResults(
            max_results=6,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )
        return Tool(
            name="web_search",
            func=search.run,
            description=(
                "Search the internet for current information, news, statistics, "
                "and research on any topic. Input: a search query string."
            ),
        )
    else:
        print("[QueryPilot] ⚠️  Web search: DuckDuckGo (set TAVILY_API_KEY for better results)")
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=6)
        search  = DuckDuckGoSearchRun(api_wrapper=wrapper)
        return Tool(
            name="web_search",
            func=search.run,
            description=(
                "Search the internet for current information, news, statistics, "
                "and research on any topic. Input: a search query string."
            ),
        )


def get_wikipedia_tool() -> Tool:
    """Returns a Wikipedia search tool for encyclopedic background knowledge."""
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=4000,
        )
    )
    return Tool(
        name="wikipedia_search",
        func=wiki.run,
        description=(
            "Search Wikipedia for background knowledge, definitions, history, "
            "and detailed explanations. Use this alongside web search. "
            "Input: a topic or concept."
        ),
    )


def get_all_tools() -> list:
    """Returns all tools available to QueryPilot."""
    return [
        get_web_search_tool(),
        get_wikipedia_tool(),
    ]
