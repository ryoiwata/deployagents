"""
Conditional Agent Example

This script demonstrates a LangGraph agent with conditional routing based on
the state's operation field.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 4_conditional_agent.py 2>/dev/null
"""

import os
from pathlib import Path
from typing import Literal, TypedDict
# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import END, START, StateGraph


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    finalNumber: int


def adder(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""
    state["finalNumber"] = state["number1"] + state["number2"]
    return state


def subtractor(state: AgentState) -> AgentState:
    """This node subtracts the 2 numbers"""
    state["finalNumber"] = state["number1"] - state["number2"]
    return state


def decide_next_node(
    state: AgentState
) -> Literal["addition_operation", "subtraction_operation"]:
    """This node will select the next node of the graph"""
    if state["operation"] == "+":
        return "addition_operation"
    elif state["operation"] == "-":
        return "subtraction_operation"


def generate_and_display_graph(
    app, output_filename: str = "conditional_agent_graph.png"
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
    graph.add_node("add_node", adder)
    graph.add_node("subtract_node", subtractor)
    graph.add_node("router", lambda state: state)  # passthrough function
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        decide_next_node,
        {
            # Edge: Node
            "addition_operation": "add_node",
            "subtraction_operation": "subtract_node"
        }
    )
    graph.add_edge("add_node", END)
    graph.add_edge("subtract_node", END)

    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app)

    # Example usage
    initial_state: AgentState = {
        "number1": 10,
        "operation": "+",
        "number2": 5,
        "finalNumber": 0
    }
    result = app.invoke(initial_state)
    print(f"Result: {result['finalNumber']}")

