"""
Automatic Higher or Lower Game (Graph V Exercise)

This script implements an automatic guessing game where the graph tries to
guess a number between 1 and 20, with a maximum of 7 attempts. The graph
uses hints to adjust its bounds and make smarter guesses.

Note: When generating the graph visualization, you may see harmless error
messages from the headless browser (Chrome/Chromium). These can be suppressed
by running the script with stderr redirection:
    python 5b_guessing_looping_agent.py 2>/dev/null
"""

import os
import random
from pathlib import Path
from typing import List, Literal, Optional, TypedDict
# framework that helps you design and manage the flow of tasks in your
# application using a graph
from langgraph.graph import END, START, StateGraph


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    player_name: str
    guesses: List[int]
    attempts: int
    lower_bound: int
    upper_bound: int
    target_number: Optional[int]  # The number to guess
    current_guess: Optional[int]  # The current guess being made
    hint: Optional[str]  # "higher", "lower", or "correct"


def setup_node(state: AgentState) -> AgentState:
    """Setup node that initializes the target number to guess"""
    # Generate a random target number between 1 and 20
    state["target_number"] = random.randint(1, 20)
    print(f"Game started! Target number is {state['target_number']} "
          f"(hidden from player)")
    return state


def guess_node(state: AgentState) -> AgentState:
    """Makes a guess based on current bounds"""
    state["attempts"] += 1

    # Make a guess using binary search approach (middle of current bounds)
    guess = (state["lower_bound"] + state["upper_bound"]) // 2
    state["current_guess"] = guess
    state["guesses"].append(guess)

    print(f"Attempt {state['attempts']}: Guessing {guess} "
          f"(bounds: {state['lower_bound']}-{state['upper_bound']})")

    return state


def hint_node(state: AgentState) -> AgentState:
    """Provides feedback and adjusts bounds based on the guess"""
    guess = state["current_guess"]
    target = state["target_number"]

    if guess == target:
        state["hint"] = "correct"
        print(f"✓ Correct! The number was {target}")
    elif guess < target:
        state["hint"] = "higher"
        # Adjust lower bound: next guess should be higher
        state["lower_bound"] = guess + 1
        print(f"  → Hint: Higher (new bounds: {state['lower_bound']}-"
              f"{state['upper_bound']})")
    else:  # guess > target
        state["hint"] = "lower"
        # Adjust upper bound: next guess should be lower
        state["upper_bound"] = guess - 1
        print(f"  → Hint: Lower (new bounds: {state['lower_bound']}-"
              f"{state['upper_bound']})")

    return state


def should_continue(
    state: AgentState
) -> Literal["continue", "exit"]:
    """Function to decide whether to continue guessing or exit"""
    # Exit if correct guess
    if state["hint"] == "correct":
        print(f"\nGame won in {state['attempts']} attempts!")
        return "exit"

    # Exit if maximum attempts reached
    if state["attempts"] >= 7:
        print(f"\nMaximum attempts ({state['attempts']}) reached. Game over!")
        print(f"The target number was {state['target_number']}")
        return "exit"

    # Continue if bounds are invalid (shouldn't happen with binary search)
    if state["lower_bound"] > state["upper_bound"]:
        print("\nBounds invalid. Exiting.")
        return "exit"

    # Continue guessing
    return "continue"


def generate_and_display_graph(
    app, output_filename: str = "guessing_looping_agent_graph.png"
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
    # Flow: setup → guess → hint → continue → (loop to guess or exit)
    graph = StateGraph(AgentState)
    graph.add_node("setup", setup_node)
    graph.add_node("guess", guess_node)
    graph.add_node("hint", hint_node)
    graph.add_node("continue", lambda state: state)  # passthrough node

    # Set entry point
    graph.add_edge(START, "setup")

    # Flow: setup → guess → hint → continue
    graph.add_edge("setup", "guess")
    graph.add_edge("guess", "hint")
    graph.add_edge("hint", "continue")

    # Conditional edge from continue: loop back to guess or exit
    graph.add_conditional_edges(
        "continue",
        should_continue,
        {
            "continue": "guess",  # Loop back to guess
            "exit": END  # Exit the game
        }
    )

    app = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(app)

    # Example usage from the exercise
    print("\n" + "="*50)
    print("Automatic Higher or Lower Game")
    print("="*50 + "\n")

    initial_state: AgentState = {
        "player_name": "Student",
        "guesses": [],
        "attempts": 0,
        "lower_bound": 1,
        "upper_bound": 20,
        "target_number": None,
        "current_guess": None,
        "hint": None
    }

    result = app.invoke(initial_state)

    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    print(f"Player: {result['player_name']}")
    print(f"Target number: {result['target_number']}")
    print(f"Total attempts: {result['attempts']}")
    print(f"Guesses made: {result['guesses']}")
    print(f"Final hint: {result['hint']}")
