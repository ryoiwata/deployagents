"""
Multiple Inputs Example

This script demonstrates a LangGraph agent that handles multiple different inputs
in the state.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python multiple_inputs.py 2>/dev/null
"""

import os
from pathlib import Path
from typing import List, TypedDict
# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import StateGraph


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


def generate_and_display_graph(
    app, output_filename: str = "multiple_inputs_graph.png"
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
    graph.add_node("processor", process_values)
    graph.set_entry_point("processor")
    graph.set_finish_point("processor")
    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app)

    # Example usage
    initial_state: AgentState = {
        "values": [1, 2, 3, 4, 5],
        "name": "Alice",
    }

    result = app.invoke(initial_state)
    print("result in main: ", result)

