"""
Hello World Agent Example

This script demonstrates a simple LangGraph agent with a greeting node.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python hello_world_agent.py 2>/dev/null
"""

import os
import subprocess
import sys
from pathlib import Path
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


def generate_and_display_graph(
    app, output_filename: str = "hello_world_agent_graph.png"
):
    """
    Generate and display the graph visualization.

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

        # Try to open the image with the system's default viewer
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(
                    ["open", str(output_path)],
                    check=False,
                    stderr=subprocess.DEVNULL
                )
            elif sys.platform == "win32":  # Windows
                subprocess.run(
                    ["start", str(output_path)],
                    shell=True,
                    check=False,
                    stderr=subprocess.DEVNULL
                )
            else:  # Linux and other Unix-like systems
                subprocess.run(
                    ["xdg-open", str(output_path)],
                    check=False,
                    stderr=subprocess.DEVNULL
                )
            print("Graph visualization opened in default viewer.")
        except Exception as e:
            print(f"Could not open image automatically: {e}")
            print(f"Please open {output_path} manually to view the graph.")
    except Exception as e:
        print(f"Warning: Could not generate graph visualization: {e}")
        print("Continuing without graph visualization...")


if __name__ == "__main__":
    # Create and configure the graph
    graph = StateGraph(AgentState)
    graph.add_node("greeter", greeting_node)
    graph.set_entry_point("greeter")
    graph.set_finish_point("greeter")
    app = graph.compile()

    # Generate and display the graph visualization
    generate_and_display_graph(app)

    # Example usage
    initial_state: AgentState = {"message": "World"}
    result = app.invoke(initial_state)
    print(result["message"])
