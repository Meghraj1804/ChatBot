# LangGraph-Based Conversational Chatbot – Workflow Overview

This project implements a state-driven conversational workflow using LangGraph, where each user interaction flows through a controlled sequence of decision-making, tool execution, and response generation. The chatbot maintains persistent conversation state and conditionally augments responses using tools and document context.

## Core Workflow Philosophy
The chatbot does not operate as a free-form agent loop.
Instead, it follows a deterministic execution graph where:

- Every user input enters through a defined start state
- Decisions are made explicitly
- Tools are invoked only when required
- The conversation exits through a defined end state

This approach ensures predictable behavior, reduced cost, and debuggable execution.

## End-to-End Conversation Workflow
### 1. User Input Initialization
- A user submits a message via the Streamlit chat interface.
- The message is wrapped as a HumanMessage.
- The message is appended to the shared ChatState.
- Each conversation is associated with a unique thread ID, enabling multi-session persistence.

### 2. Entry Into LangGraph (START)
- The updated ChatState is passed to the LangGraph workflow.
- The workflow begins execution from the START node.
- The state contains the full message history for the current thread.

### 3. Decision Node (Intent & Tool Requirement)
- A dedicated decision model analyzes the latest user message.
- The model determines whether:
    - A direct LLM response is sufficient, or
    - One or more external tools are required
- The decision output controls the next transition in the graph.
- Outcome:
    - If no tools are needed → route to response generation
    - If tools are required → route to tool execution

### 4. Conditional Tool Invocation
This step enriches the conversation state with external data while keeping execution controlled.
- When tools are required, execution flows to the Tool Node.
- Tools are invoked using LangChain’s structured tool interface.
- Available tools include:
    - Web search
    - Currency conversion
- Tool results are returned as ToolMessage objects and appended to the ChatState.

### 5. Response Generation Node
- The generation model receives:
    - Full conversation history
    - Tool outputs (if any)
- Retrieved document context (when available)
- The LLM generates a final response grounded in the updated state.
- The response is appended as an AIMessage.

### 6. Streaming Output to UI
- The generated response is streamed incrementally back to the Streamlit UI.
- This improves perceived responsiveness while maintaining backend determinism.
- Only AI-generated content is streamed; state updates remain internal.

### 7. Exit From Graph (END)
- Once response generation completes, execution transitions to the END node.
- The final ChatState is checkpointed.
- The conversation thread remains available for future continuation.

## Document-Aware Workflow (When PDFs Are Used)
When documents are uploaded:
1. PDFs are loaded and split into chunks.
2. Embeddings are generated for each chunk.
3. Vectors are stored persistently using FAISS.
4. During response generation:
    - Relevant chunks are retrieved
    - Retrieved context is injected into the LLM input
5. The chatbot produces context-aware answers grounded in uploaded documents.

## Conversation Persistence Workflow
This enables long-running, multi-turn interactions without losing context.
- Each chat session is associated with a unique thread ID.
- Conversation state is stored using LangGraph checkpointing.
- Users can:
    - Resume previous conversations
    - Switch between threads
    - Reset sessions without losing stored data

## Features
- Deterministic execution over probabilistic agent loops
- Controlled tool usage to reduce unnecessary calls
- Clear separation between decision-making and response generation
- Persistent state for realistic conversational behavior
- Scalable foundation for production hardening