import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.prebuilt import InjectedState, ToolNode  # noqa: E402
from src.utils.visualize import generate_and_display_graph  # noqa: E402

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str
    working_directory: str


@tool
def update(
    content: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Updates the document with the provided content."""
    state["document_content"] = content
    return (
        f"Document has been updated successfully! The current content "
        f"is:\n{content}"
    )


@tool
def save(
    filename: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """
    document_content = state.get("document_content", "")
    working_directory = state.get("working_directory", "")

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    # Use working_directory if provided, otherwise use current directory
    if working_directory:
        # Ensure the working directory exists
        work_dir = Path(working_directory)
        work_dir.mkdir(parents=True, exist_ok=True)
        filepath = work_dir / filename
    else:
        filepath = Path(filename)

    try:
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(document_content)

        # Get absolute path for display
        abs_path = filepath.resolve()
        print(f"\n💾 Document has been saved to: {abs_path}")
        return f"Document has been saved successfully to '{abs_path}'."

    except Exception as e:
        error_msg = f"Error saving document: {str(e)}"
        print(f"\n❌ {error_msg}")
        return error_msg


tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    document_content = state.get("document_content", "")
    system_prompt = SystemMessage(
        content=(
            "You are Drafter, a helpful writing assistant. You are going to "
            "help the user update and modify documents.\n"
            "- If the user wants to update or modify content, use the "
            "'update' tool with the complete updated content.\n"
            "- If the user wants to save and finish, you need to use the "
            "'save' tool.\n"
            "- Make sure to always show the current document state after "
            "modifications.\n"
            f"The current document content is:\n{document_content}"
        )
    )

    if not state["messages"]:
        user_input = (
            "I'm ready to help you update a document. "
            "What would you like to create?"
        )
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input(
            "\nWhat would you like to do with the document? "
        )
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        print(f"🔧 USING TOOLS: {tool_names}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def process_tool_results(state: AgentState) -> AgentState:
    """Process tool results and update state based on tool calls."""
    import json
    messages = state["messages"]

    # Find the last agent message with tool calls
    last_agent_message = None
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            last_agent_message = msg
            break

    if last_agent_message and last_agent_message.tool_calls:
        for tool_call in last_agent_message.tool_calls:
            tool_name = tool_call["name"]

            if tool_name == "update":
                # Extract content from tool call arguments
                if isinstance(tool_call.get("args"), str):
                    args = json.loads(tool_call["args"])
                else:
                    args = tool_call.get("args", {})
                content = args.get("content", "")
                state["document_content"] = content

    return state


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]

    if not messages:
        return "continue"

    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and
                "saved" in message.content.lower() and
                "document" in message.content.lower()):
            return "end"  # goes to the end edge which leads to the endpoint

    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("process_tools", process_tool_results)

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")
graph.add_edge("tools", "process_tools")

graph.add_conditional_edges(
    "process_tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

# Generate and save the graph visualization
generate_and_display_graph(app, "drafter_agent_graph.png")


def run_document_agent():
    print("\n ===== DRAFTER =====")

    state = {
        "messages": [],
        "document_content": "",
        "working_directory": "",
    }

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
