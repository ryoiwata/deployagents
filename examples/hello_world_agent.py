"""
Hello World Agent Example

This script demonstrates a simple LangGraph agent with a greeting node.
"""

from typing import TypedDict
from langgraph.graph import StateGraph  # framework that helps you design and manage the flow of tasks in your application using a graph


# We now create an AgentState - shared data structure that keeps track of information as your application runs.
class AgentState(TypedDict):
    message: str


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""
    state['message'] = "Hey " + state["message"] + ", how is your day going?"
    return state


if __name__ == "__main__":
    # Example usage
    initial_state: AgentState = {"message": "World"}
    result = greeting_node(initial_state)
    print(result["message"])

