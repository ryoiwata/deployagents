"""
Hello World Agent Example

This script demonstrates a simple LangGraph agent with a greeting node.
"""

from typing import TypedDict
# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import StateGraph


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

    # Example usage
    initial_state: AgentState = {"message": "World"}
    result = app.invoke(initial_state)
    print(result["message"])
