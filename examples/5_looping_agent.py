"""
Looping Agent Example

This script demonstrates a LangGraph agent with looping logic that continues
until a counter reaches a threshold.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 5_looping_agent.py 2>/dev/null
"""

import os
import random
from pathlib import Path
from typing import List, Literal, TypedDict
# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import END, StateGraph


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


def generate_and_display_graph(
    app, output_filename: str = "looping_agent_graph.png"
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
    generate_and_display_graph(app)

    # Example usage
    initial_state: AgentState = {
        "name": "Alice",
        "counter": 0,
        "number": []
    }
    result = app.invoke(initial_state)
    print(f"\nFinal state:")
    print(f"Name: {result['name']}")
    print(f"Counter: {result['counter']}")
    print(f"Random numbers: {result['number']}")

