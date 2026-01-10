"""
Agent Bot Example

This script demonstrates a LangGraph agent bot that uses OpenAI's ChatOpenAI
to process and respond to messages.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 6_agent_bot.py 2>/dev/null

Note: This script requires an OpenAI API key set in your environment variables
or a .env file. Make sure to set OPENAI_API_KEY before running.
"""

import os
from pathlib import Path
from typing import List, TypedDict

from dotenv import load_dotenv  # used to store secret stuff like API keys
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI chat model
llm = ChatOpenAI(model="gpt-4o")


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    messages: List[HumanMessage]


def process(state: AgentState) -> AgentState:
    """Process messages using OpenAI"""
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    state["messages"].append(response)
    return state


def generate_and_display_graph(
    app, output_filename: str = "agent_bot_graph.png"
):
    """
    Generate and save the graph visualization to a PNG file.

    Args:
        app: The compiled LangGraph application
        output_filename: Name of the output PNG file

    Note: Browser-related error messages may appear but are harmless.
    """
    print("Generating graph visualization...")
    try:
        # Set environment variables to minimize Chrome/Chromium error output
        original_env = os.environ.copy()
        os.environ["PYTHONWARNINGS"] = "ignore"
        # Suppress Chrome logging
        os.environ["CHROME_LOG_FILE"] = "/dev/null"

        # Generate the graph PNG (may show harmless browser errors in stderr)
        graph_png = app.get_graph().draw_mermaid_png()

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

        output_path = Path(__file__).parent / output_filename
        with open(output_path, "wb") as f:
            f.write(graph_png)
        print(f"Graph visualization saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not generate graph visualization: {e}")
        print("Continuing without graph visualization...")


if __name__ == "__main__":
    # Create and configure the graph
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    agent = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(agent)

    # Example usage - interactive input with continuous conversation
    print("\n" + "="*50)
    print("Agent Bot Example")
    print("="*50 + "\n")

    user_input = input("Enter: ")
    while user_input != "exit":
        agent.invoke({"messages": [HumanMessage(content=user_input)]})
        user_input = input("Enter: ")
