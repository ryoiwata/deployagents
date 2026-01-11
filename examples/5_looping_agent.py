"""
Looping Agent Example

This script demonstrates a LangGraph agent with looping logic that continues
until a counter reaches a threshold.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 5_looping_agent.py 2>/dev/null
"""

import sys
from pathlib import Path
import random
from typing import List, Literal, TypedDict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import END, StateGraph  # noqa: E402
from src.utils.visualize import generate_and_display_graph  # noqa: E402


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    name: str
    counter: int
    number: List[int]


def greeting_node(state: AgentState) -> AgentState:
    """Greeting Node which says hi to the person"""
    state["name"] = f"Hi there, {state['name']}"
    state["counter"] = 0
    return state


def random_node(state: AgentState) -> AgentState:
    """Generates a random number from 0 to 10"""
    state["number"].append(random.randint(0, 10))
    state["counter"] += 1
    return state


def should_continue(
    state: AgentState
) -> Literal["loop", "exit"]:
    """Function to decide what to do next"""
    if state["counter"] < 5:
        print("ENTERING LOOP", state["counter"])
        return "loop"  # Continue looping
    else:
        return "exit"  # Exit the loop


if __name__ == "__main__":
    # Create and configure the graph
    # Flow: greeting → random → random → random → random → random → END
    graph = StateGraph(AgentState)
    graph.add_node("greeting", greeting_node)
    graph.add_node("random", random_node)
    graph.add_edge("greeting", "random")
    graph.add_conditional_edges(
        "random",  # Source node
        should_continue,  # Action (a function that determines the next step)
        {
            "loop": "random",  # If "should_continue" returns "loop", go back
            # to "random" (self-loop)
            "exit": END  # If "should_continue" returns "exit", end the graph
        }
    )
    graph.set_entry_point("greeting")

    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app, "looping_agent_graph.png")

    # Example usage
    initial_state: AgentState = {
        "name": "Alice",
        "counter": 0,
        "number": []
    }
    result = app.invoke(initial_state)
    print("\nFinal state:")
    print(f"Name: {result['name']}")
    print(f"Counter: {result['counter']}")
    print(f"Random numbers: {result['number']}")
