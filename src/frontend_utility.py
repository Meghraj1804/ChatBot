from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import streamlit as st
from src.backend import chatbot

def get_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = get_thread_id()
    st.session_state['current_thread_id'] = thread_id
    st.session_state['message_history'] = []
    if thread_id not in st.session_state['thread_messages']:
        st.session_state['thread_messages'].append(thread_id)
    return thread_id

def load_conversations():
    messages = chatbot.get_state(config = {'configurable': {'thread_id': st.session_state['current_thread_id']}})
    return messages.values.get('messages',[])



def ai_only_stream(user_input, context=None, metadata=None):
    status_holder = {"box": None}
    full_response = ""    

    for message_chunk, metadata in chatbot.stream(
                                                {'messages':[HumanMessage(content=user_input)]},
                                                config = {
                                                "configurable": {"thread_id": st.session_state["current_thread_id"]},
                                                "metadata": {"thread_id": st.session_state["current_thread_id"]},
                                                "run_name": "chat_turn",
                                                },
                                                stream_mode='messages'
                                                ):

        if isinstance(message_chunk, ToolMessage):
            tool_name = getattr(message_chunk, "name", "tool")

            if status_holder["box"] is None:
                status_holder["box"] = st.status(
                    f"🔧 Using `{tool_name}` …",
                    expanded=True
                )
            else:
                status_holder["box"].update(
                    label=f"🔧 Using `{tool_name}` …",
                    state="running",
                    expanded=True,
                )

        if metadata.get('langgraph_node') == 'decision_node':
            continue

        if isinstance(message_chunk, AIMessage) and message_chunk.content:
            full_response += message_chunk.content
            yield message_chunk.content

    # close once, at the end
    if status_holder["box"] is not None:
        status_holder["box"].update(
            label="✅ Tool finished",
            state="complete",
            expanded=False
        )