import sys
from pathlib import Path
from typing import List, TypedDict, Union

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# used to store secret stuff like API keys
from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from src.utils.visualize import generate_and_display_graph  # noqa: E402

# Load environment variables from .env file
load_dotenv()


# We now create an AgentState - shared data structure that keeps track of
# information as your application runs.
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


# Initialize the OpenAI chat model
llm = ChatOpenAI(model="gpt-4o")


def process(state: AgentState) -> AgentState:
    """Process messages using OpenAI"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))

    print(f"\nAI: {response.content}")
    print("Current state: ", state["messages"])

    return state


if __name__ == "__main__":
    # Create and configure the graph
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    agent = graph.compile()

    # Generate and save the graph visualization
    generate_and_display_graph(agent, "memory_agent_graph.png")

    # Example usage - interactive input with continuous conversation
    print("\n" + "="*50)
    print("Agent Bot Example")
    print("="*50 + "\n")

    conversation_history = []

    user_input = input("Enter: ")
    while user_input != "exit":
        conversation_history.append(HumanMessage(content=user_input))

        result = agent.invoke({"messages": conversation_history})

        conversation_history = result["messages"]

        user_input = input("Enter: ")
