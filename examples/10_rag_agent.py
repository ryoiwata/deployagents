# Note: Install required packages with:
# pip install langchain-text-splitters
from dotenv import load_dotenv
import os
from pathlib import Path
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage
)
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# I want to minimize hallucination - temperature = 0 makes the model
# output more deterministic
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Our Embedding Model - has to also be compatible with the LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Global retriever that will be set when vectorstore is initialized
retriever = None
document_name = None


def initialize_vectorstore(pdf_path: str):
    """Initialize the vectorstore from a PDF file.

    Args:
        pdf_path: Path to the PDF file to load

    Returns:
        The retriever object for the vectorstore
    """
    global retriever, document_name

    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Extract document name from PDF path for collection naming
    pdf_file = Path(pdf_path)
    document_name = pdf_file.stem  # Get filename without extension

    print(f"Loading PDF: {pdf_path}")

    # Load the PDF
    pdf_loader = PyPDFLoader(pdf_path)
    try:
        pages = pdf_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise

    # Chunking Process
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    pages_split = text_splitter.split_documents(pages)

    # Generate persist_directory and collection_name from PDF filename
    # Use a safe directory name based on the PDF filename
    base_dir = Path(__file__).parent / "vectorstores"
    persist_directory = str(base_dir / document_name)
    collection_name = document_name.lower().replace(" ", "_")

    # Create directory if it doesn't exist
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    try:
        # Create the chroma database using our embedding model
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"Created ChromaDB vector store for '{document_name}'!")

    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        raise

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # K is the amount of chunks to return
    )

    return retriever


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns relevant information from the loaded
    document.
    Use this tool to retrieve specific information from the document when
    answering questions.
    """
    global retriever, document_name

    if retriever is None:
        return (
            "Error: No document has been loaded. "
            "Please initialize the vectorstore first."
        )

    docs = retriever.invoke(query)

    doc_ref = document_name if document_name else "the document"
    if not docs:
        return f"I found no relevant information in {doc_ref}."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


def get_system_prompt():
    """Generate a generic system prompt for the RAG agent."""
    return """
You are an intelligent AI assistant who answers questions based on the
PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the document
content. You can make multiple calls if needed.
If you need to look up some information before asking a follow up
question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your
answers.
"""


# Creating a dictionary of our tools
tools_dict = {our_tool.name: our_tool for our_tool in tools}


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=get_system_prompt())] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        query = t['args'].get('query', 'No query provided')
        print(f"Calling Tool: {t['name']} with query: {query}")

        # Checks if a valid tool is present
        if t['name'] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = (
                "Incorrect Tool Name, Please Retry and Select tool "
                "from List of Available tools."
            )
        else:
            result = tools_dict[t['name']].invoke(
                t['args'].get('query', '')
            )
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            )
        )

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    """Main function to run the RAG agent with user-specified PDF."""
    print("\n=== RAG AGENT ===")

    # Ask user for PDF file path
    while True:
        pdf_path = input("\nEnter the path to the PDF file: ").strip()

        # Remove quotes if user pasted a path with quotes
        pdf_path = pdf_path.strip('"').strip("'")

        if not pdf_path:
            print("Please provide a valid PDF file path.")
            continue

        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"Error: File not found: {pdf_path}")
            retry = input(
                "Would you like to try another path? (y/n): "
            ).strip().lower()
            if retry != 'y':
                print("Exiting...")
                return
            continue

        # Check if it's a PDF file
        if not pdf_path.lower().endswith('.pdf'):
            print("Warning: The file doesn't have a .pdf extension.")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                continue

        # Initialize vectorstore
        try:
            print(f"\nInitializing vectorstore for: {pdf_path}")
            initialize_vectorstore(pdf_path)
            print("Vectorstore initialized successfully!\n")
            break
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
            retry = input(
                "Would you like to try another file? (y/n): "
            ).strip().lower()
            if retry != 'y':
                print("Exiting...")
                return

    # Enter chat loop
    print("You can now ask questions about the document.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        user_input = input("\nWhat is your question: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break

        if not user_input:
            continue

        messages = [HumanMessage(content=user_input)]

        try:
            result = rag_agent.invoke({"messages": messages})

            print("\n=== ANSWER ===")
            print(result['messages'][-1].content)
        except Exception as e:
            print(f"\nError processing question: {e}")


if __name__ == "__main__":
    running_agent()
