"""
ReAct Agent Example

This script demonstrates a ReAct (Reasoning + Acting) agent using LangGraph
with tool calling capabilities.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 8_ReAct.py 2>/dev/null

Note: This script requires an OpenAI API key set in your environment variables
or a .env file. Make sure to set OPENAI_API_KEY before running.
"""

import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402
# The foundational class for all message types in LangGraph
from langchain_core.messages import BaseMessage  # noqa: E402
# Passes data back to LLM after it calls a tool such as the content and the
from langchain_core.messages import ToolMessage  # noqa: E402
# Message for providing instructions to the LLM
from langchain_core.messages import SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.graph import StateGraph, END  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from src.utils.visualize import generate_and_display_graph  # noqa: E402

load_dotenv()
