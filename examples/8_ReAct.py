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
from langchain_core.messages import BaseMessage, HumanMessage  # noqa: E402
# Message for providing instructions to the LLM
from langchain_core.messages import SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.graph import StateGraph, END  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from src.utils.visualize import generate_and_display_graph  # noqa: E402

load_dotenv()


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b


tools = [add]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    """Model call function that invokes the AI model"""
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best "
        "of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Function to decide whether to continue or end"""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def print_stream(stream):
    """Print messages from the stream"""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    # Create and configure the graph
    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call)

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("our_agent")

    graph.add_conditional_edges(
        "our_agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    graph.add_edge("tools", "our_agent")

    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app, "react_agent_graph.png")

    # Example usage
    print("\n" + "="*50)
    print("ReAct Agent Example")
    print("="*50 + "\n")

    # Get user input
    user_input = input("Enter your query: ")

    inputs = {
        "messages": [
            SystemMessage(
                content="You are my AI assistant, please answer my query to "
                "the best of your ability."
            ),
            HumanMessage(content=user_input)
        ]
    }

    # Stream the execution
    print("\nStreaming agent execution...\n")
    print_stream(app.stream(inputs, stream_mode="values"))
