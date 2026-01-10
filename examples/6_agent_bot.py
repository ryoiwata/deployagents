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


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    messages: List[HumanMessage]


def agent_node(state: AgentState) -> AgentState:
    """Agent node that processes messages using OpenAI"""
    # Initialize the OpenAI chat model
    model = ChatOpenAI(model="gpt-3.5-turbo")

    # Get the last message from the state
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if last_message:
        # Invoke the model with the message
        response = model.invoke([last_message])
        # Add the response to messages
        state["messages"].append(response)
        print(f"Agent response: {response.content}")
    else:
        print("No messages to process")

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
    graph.add_node("agent", agent_node)

    # Set entry point and flow
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app)

    # Example usage
    print("\n" + "="*50)
    print("Agent Bot Example")
    print("="*50 + "\n")

    initial_state: AgentState = {
        "messages": [
            HumanMessage(content="Hello! What is the capital of France?")
        ]
    }

    result = app.invoke(initial_state)

    print("\n" + "="*50)
    print("Conversation:")
    print("="*50)
    for i, msg in enumerate(result["messages"], 1):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "Agent"
        print(f"{msg_type} {i}: {msg.content}")
