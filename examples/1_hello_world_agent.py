"""
Hello World Agent Example

This script demonstrates a simple LangGraph agent with a greeting node.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python hello_world_agent.py 2>/dev/null
"""

import sys
from pathlib import Path
from typing import TypedDict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import StateGraph  # noqa: E402
from src.utils.visualize import generate_and_display_graph  # noqa: E402


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    message: str


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""
    state['message'] = "Hey " + state["message"] + ", how is your day going?"
    return state


if __name__ == "__main__":
    # Create and configure the graph
    graph = StateGraph(AgentState)
    graph.add_node("greeter", greeting_node)
    graph.set_entry_point("greeter")
    graph.set_finish_point("greeter")
    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app, "hello_world_agent_graph.png")

    # Example usage
    initial_state: AgentState = {"message": "World"}
    result = app.invoke(initial_state)
    print(result["message"])
