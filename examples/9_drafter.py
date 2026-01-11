"""
Drafter Agent Example

This script demonstrates a drafter agent using LangGraph with tool calling
capabilities for document management.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 9_drafter.py 2>/dev/null

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
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from src.utils.visualize import generate_and_display_graph  # noqa: E402

load_dotenv()

# This is the global variable to store document content
document_content = ""
