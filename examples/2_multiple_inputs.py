"""
Multiple Inputs Example

This script demonstrates a LangGraph agent that handles multiple different inputs
in the state.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python multiple_inputs.py 2>/dev/null
"""

import sys
from pathlib import Path
from typing import List, TypedDict

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
    values: List[int]
    name: str
    result: str


def process_values(state: AgentState) -> AgentState:
    """This function handles multiple different inputs"""
    
    print("state in process_values before", state)

    state["result"] = (
        f"Hi there {state['name']}! Your sum = {sum(state['values'])}"
    )
    
    print("state in process_values after", state)

    return state


if __name__ == "__main__":
    # Create and configure the graph
    graph = StateGraph(AgentState)
    graph.add_node("processor", process_values)
    graph.set_entry_point("processor")
    graph.set_finish_point("processor")
    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app, "multiple_inputs_graph.png")

    # Example usage
    initial_state: AgentState = {
        "values": [1, 2, 3, 4, 5],
        "name": "Alice",
    }

    result = app.invoke(initial_state)
    print("result in main: ", result)

