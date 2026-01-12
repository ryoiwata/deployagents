import json
import logging
import os
import sys
from datetime import datetime
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

# Initialize logger that writes to timestamped log file in logs/ directory
logger = logging.getLogger('agent_debug')
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
logger.handlers = []

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)

# Generate timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"agent_session_{timestamp}.log"
log_file = logs_dir / log_filename

# Create file handler
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter (we'll format as JSON in the log statements)
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.propagate = False


def create_log_entry(
    step: int,
    current_node: str,
    state: dict,
    tool_execution: dict = None,
    state_snapshot: dict = None
) -> dict:
    """Create a structured log entry with timestamp and all required fields."""
    timestamp = datetime.now().isoformat()

    document_content = state.get("document_content", "")
    messages = state.get("messages", [])

    # Get the latest message for logging
    latest_message_data = None
    if messages:
        latest_message = messages[-1]
        if hasattr(latest_message, 'content'):
            content_preview = (
                latest_message.content[:500]
                if latest_message.content else ""
            )
            latest_message_data = {
                "type": type(latest_message).__name__,
                "content": content_preview,
            }
            # Add tool_calls if present
            if (hasattr(latest_message, 'tool_calls') and
                    latest_message.tool_calls):
                latest_message_data["tool_calls"] = [
                    {
                        "name": tc.get("name", ""),
                        "args": str(tc.get("args", ""))[:200]
                    }
                    for tc in latest_message.tool_calls
                ]
        else:
            latest_message_data = {
                "type": type(latest_message).__name__,
                "content": str(latest_message)[:500]
            }

    doc_preview = (
        document_content[:200] if document_content else ""
    )
    log_entry = {
        "timestamp": timestamp,
        "step": step,
        "current_node": current_node,
        "document_content_length": len(document_content),
        "document_content_preview": doc_preview,
        "total_messages": len(messages),
        "latest_message": latest_message_data,
        "working_directory": state.get("working_directory", ""),
    }

    # Add tool execution details if provided
    if tool_execution:
        log_entry["tool_execution"] = tool_execution

    # Add state snapshot if provided
    if state_snapshot:
        log_entry["state_snapshot"] = state_snapshot

    return log_entry


def log_entry(log_entry_dict: dict):
    """Log a structured entry as pretty-printed JSON."""
    logger.info(json.dumps(log_entry_dict, indent=2))


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str
    working_directory: str


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    # Return success message
    # State will be updated via process_tool_results node
    return (
        f"Document has been updated successfully! The current content "
        f"is:\n{content}"
    )


@tool
def save(filename: str, state: Annotated[dict, InjectedState]) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """
    # Retrieve document_content from injected state dictionary
    document_content = state.get("document_content", "")
    working_directory = state.get("working_directory", "")

    # Sanity check: print the length of content for debugging
    content_length = len(document_content) if document_content else 0
    print(f"\n🔍 DEBUG: Document content length: {content_length} characters")

    # Check if document_content is not empty before attempting to write
    if not document_content:
        error_msg = "Cannot save: document content is empty."
        print(f"\n❌ {error_msg}")
        return error_msg

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    # Path safety: Use os.path.join to ensure the file is saved correctly
    if working_directory:
        # Ensure the working directory exists
        os.makedirs(working_directory, exist_ok=True)
        filepath = os.path.join(working_directory, filename)
    else:
        filepath = filename

    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Write the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(document_content)

        # Get absolute path for display
        abs_path = os.path.abspath(filepath)
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
    """Process tool results and update state using functional pattern."""
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
                # Return state update dictionary (functional pattern)
                return {"document_content": content}

    # No state update needed
    return {}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation.

    Checks the last message in the state: if it's a ToolMessage indicating
    a successful save, return 'end'; otherwise, return 'agent' to continue.
    """
    messages = state["messages"]

    if not messages:
        return "agent"

    # Check the last message
    last_message = messages[-1]

    # If it's a ToolMessage indicating successful save, end the conversation
    if (isinstance(last_message, ToolMessage) and
            "saved" in last_message.content.lower() and
            "document" in last_message.content.lower() and
            "successfully" in last_message.content.lower()):
        return "end"

    # Otherwise, continue to agent
    return "agent"


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
graph.add_node("process_tool_results", process_tool_results)

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")
graph.add_edge("tools", "process_tool_results")

graph.add_conditional_edges(
    "process_tool_results",
    should_continue,
    {
        "agent": "agent",
        "end": END,
    },
)

app = graph.compile()

# Generate and save the graph visualization
generate_and_display_graph(app, "drafter_agent_graph.png")


def run_document_agent():
    print("\n ===== DRAFTER =====")

    initial_state = {
        "messages": [],
        "document_content": "",
        "working_directory": "",
    }

    step_count = 0
    # Track accumulated state by merging updates
    accumulated_state = initial_state.copy()

    # Use stream_mode="updates" to get node-level information
    stream = app.stream(initial_state, stream_mode="updates")
    for update in stream:
        step_count += 1
        # Extract node name and update from the dictionary
        node_name, node_update = next(iter(update.items()))

        # Skip if node_update is None
        if node_update is None:
            continue

        # Store state before this node's update (for snapshots)
        state_before = accumulated_state.copy()

        # Merge the node update into accumulated state
        # For messages, we need to handle the add_messages reducer
        if "messages" in node_update:
            # Messages are appended by the add_messages reducer
            existing_msgs = accumulated_state.get("messages", [])
            new_msgs = node_update["messages"]
            accumulated_state["messages"] = (
                list(existing_msgs) + list(new_msgs)
            )

        # For other fields, just update them
        for key, value in node_update.items():
            if key != "messages":
                accumulated_state[key] = value

        # Current state after this node's update
        current_state = accumulated_state.copy()

        # Extract tool execution details if this is the tools node
        tool_execution = None
        if node_name == "tools":
            # Find tool messages that were just added
            messages = current_state.get("messages", [])
            tool_messages = [
                msg for msg in messages if isinstance(msg, ToolMessage)
            ]

            if tool_messages:
                # Get the most recent tool message
                latest_tool_msg = tool_messages[-1]

                # Find the corresponding tool call from previous agent message
                tool_call_info = None
                for msg in reversed(messages):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # Find matching tool call
                        for tc in msg.tool_calls:
                            if tc.get("id") == latest_tool_msg.tool_call_id:
                                tool_call_info = tc
                                break
                        if tool_call_info:
                            break

                if tool_call_info:
                    tool_name = tool_call_info.get("name", "")
                    tool_args = tool_call_info.get("args", {})
                    tool_response = latest_tool_msg.content

                    tool_execution = {
                        "tool_name": tool_name,
                        # Truncate long args
                        "tool_args": str(tool_args)[:500],
                        # Truncate long responses
                        "tool_response": (
                            tool_response[:500] if tool_response else ""
                        )
                    }

        # Track state snapshot for process_tool_results node
        state_snapshot = None
        if node_name == "process_tool_results":
            # Capture state before and after
            doc_before = state_before.get("document_content", "")
            doc_after = current_state.get("document_content", "")

            state_snapshot = {
                "before": {
                    "document_content": doc_before,
                    "document_content_length": len(doc_before)
                },
                "after": {
                    "document_content": doc_after,
                    "document_content_length": len(doc_after)
                }
            }

        # Create log entry
        log_entry_dict = create_log_entry(
            step=step_count,
            current_node=node_name,
            state=current_state,
            tool_execution=tool_execution,
            state_snapshot=state_snapshot
        )

        # Log the entry
        log_entry(log_entry_dict)

        # Print messages for console output
        if "messages" in current_state:
            print_messages(current_state["messages"])

    print("\n ===== DRAFTER FINISHED =====")
    finish_log = {
        "timestamp": datetime.now().isoformat(),
        "event": "agent_finished",
        "total_steps": step_count
    }
    log_entry(finish_log)


if __name__ == "__main__":
    run_document_agent()
