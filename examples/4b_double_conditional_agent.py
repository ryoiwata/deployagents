"""
Double Conditional Agent Example (Graph IV)

This script demonstrates a LangGraph agent with two stages of conditional
routing based on the state's operation and operation2 fields.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 4b_souble_conditional_agent.py 2>/dev/null
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
    number3: int
    number4: int
    operation2: str
    finalNumber: int
    finalNumber2: int


def adder(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""
    state["finalNumber"] = state["number1"] + state["number2"]
    return state


def subtractor(state: AgentState) -> AgentState:
    """This node subtracts the 2 numbers"""
    state["finalNumber"] = state["number1"] - state["number2"]
    return state


def adder2(state: AgentState) -> AgentState:
    """This node adds number3 and number4"""
    state["finalNumber2"] = state["number3"] + state["number4"]
    return state


def subtractor2(state: AgentState) -> AgentState:
    """This node subtracts number4 from number3"""
    state["finalNumber2"] = state["number3"] - state["number4"]
    return state


def decide_next_node(
    state: AgentState
) -> Literal["addition_operation", "subtraction_operation"]:
    """This node will select the next node of the graph (first router)"""
    if state["operation"] == "+":
        return "addition_operation"
    elif state["operation"] == "-":
        return "subtraction_operation"


def decide_next_node2(
    state: AgentState
) -> Literal["addition_operation2", "subtraction"]:
    """This node will select the next node of the graph (second router)"""
    if state["operation2"] == "+":
        return "addition_operation2"
    elif state["operation2"] == "-":
        return "subtraction"


def generate_and_display_graph(
    app, output_filename: str = "double_conditional_agent_graph.png"
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
    # Create and configure the graph with two conditional edges
    graph = StateGraph(AgentState)

    # First stage nodes
    graph.add_node("add_node", adder)
    graph.add_node("subtract_node", subtractor)

    # Second stage nodes
    graph.add_node("add_node2", adder2)
    graph.add_node("subtract_node2", subtractor2)

    # Router nodes
    graph.add_node("router", lambda state: state)  # passthrough function
    graph.add_node("router2", lambda state: state)  # passthrough function

    # Entry point
    graph.add_edge(START, "router")

    # First conditional edge from router
    graph.add_conditional_edges(
        "router",
        decide_next_node,
        {
            # Edge: Node
            "addition_operation": "add_node",
            "subtraction_operation": "subtract_node"
        }
    )

    # Both first stage nodes converge to router2
    graph.add_edge("add_node", "router2")
    graph.add_edge("subtract_node", "router2")

    # Second conditional edge from router2
    graph.add_conditional_edges(
        "router2",
        decide_next_node2,
        {
            "addition_operation2": "add_node2",
            "subtraction": "subtract_node2"
        }
    )

    # Both second stage nodes converge to END
    graph.add_edge("add_node2", END)
    graph.add_edge("subtract_node2", END)

    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app)

    # Example usage from the exercise
    initial_state: AgentState = {
        "number1": 10,
        "operation": "-",
        "number2": 5,
        "number3": 7,
        "number4": 2,
        "operation2": "+",
        "finalNumber": 0,
        "finalNumber2": 0
    }
    result = app.invoke(initial_state)
    print(f"First operation result: {result['finalNumber']}")
    print(f"Second operation result: {result['finalNumber2']}")
