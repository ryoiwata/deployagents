"""
Conditional Agent Example

This script demonstrates a LangGraph agent with conditional routing based on
the state's operation field.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 4_conditional_agent.py 2>/dev/null
"""

import sys
from pathlib import Path
from typing import Literal, TypedDict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import END, START, StateGraph
from src.utils.visualize import generate_and_display_graph


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
    generate_and_display_graph(app, "conditional_agent_graph.png")

    # Example usage
    initial_state: AgentState = {
        "number1": 10,
        "operation": "+",
        "number2": 5,
        "finalNumber": 0
    }
    result = app.invoke(initial_state)
    print(f"Result: {result['finalNumber']}")

