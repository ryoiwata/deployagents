"""
Sequential Agent Example

This script demonstrates a LangGraph agent with sequential nodes that process
state in a defined order.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 3_sequential_agent.py 2>/dev/null
"""

import sys
from pathlib import Path
from typing import TypedDict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import StateGraph
from src.utils.visualize import generate_and_display_graph


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    name: str
    age: str
    final: str


def first_node(state: AgentState) -> AgentState:
    """This is the first node of our sequence"""
    
    print("state in first_node before", state)
    state["final"] = f"Hi {state['name']}!"
    print("state in first_node after", state)

    return state


def second_node(state: AgentState) -> AgentState:
    """This is the second node of our sequence"""
    
    print("state in second_node before", state)
    state["final"] = state["final"] + f"You are {state['age']} years old!"
    print("state in second_node after", state)
    return state


if __name__ == "__main__":
    # Create and configure the graph
    graph = StateGraph(AgentState)
    graph.add_node("first_node", first_node)
    graph.add_node("second_node", second_node)
    graph.set_entry_point("first_node")
    graph.add_edge("first_node", "second_node")
    graph.set_finish_point("second_node")
    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app, "sequential_agent_graph.png")

    # Example usage
    initial_state: AgentState = {
        "name": "Alice",
        "age": "25",
    }
    print("initial_state in main: ", initial_state)
    result = app.invoke(initial_state)
    print("result in main: ", result)
    print(result["final"])

