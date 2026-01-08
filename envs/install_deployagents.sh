#!/bin/bash
# obsynapse Environment Setup Script
# This script creates a conda environment with all dependencies needed for the obsynapse project

# Deactivate any currently active conda environment
conda deactivate

# Create conda environment with Python 3.12 (compatible with DBT)
# Using local path ./deployagents instead of global environment
conda create -p ./deployagents python=3.12 --yes

# Activate the newly created environment
conda activate ./deployagents

# langchain: Framework for building LLM applications
# Used for text splitting (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter)
# and integration with vector stores for the hierarchical chunking pipeline
conda install conda-forge::langchain --yes

# langgraph: Library for building stateful, multi-actor applications with LLMs
# Used for orchestrating the flashcard generation workflow (Generator-Critic loop)
# and managing state transitions in the Agentic Forge pipeline
conda install conda-forge::langgraph --yes

