from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import streamlit as st
from src.backend import chatbot
from src.exception import CustomException
from src.logger import logging
import sys

def get_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    try:
        thread_id = str(get_thread_id())
        st.session_state['current_thread_id'] = thread_id
        st.session_state['message_history'] = []
        if thread_id not in st.session_state['thread_messages']:
            st.session_state['thread_messages'].append(thread_id)
        st.session_state['thread_docs_history'] = {thread_id: {'retriever': None, 'documents':[]}}
        return thread_id
    except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

def load_conversations():
    try:
        messages = chatbot.get_state(config = {'configurable': {'thread_id': st.session_state['current_thread_id']}})
        return messages.values.get('messages',[])
    except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

def ai_only_stream(user_input, context=None, metadata=None):
    try:
        status_holder = {"box": None}
        full_response = ""
        
        for message_chunk, metadata in chatbot.stream(
                                                    {'messages':[HumanMessage(content=user_input)]},
                                                    config = {
                                                    "configurable": {"thread_id": st.session_state["current_thread_id"],
                                                                    "user_id": st.session_state["user_id"],
                                                                    "docs":st.session_state['thread_docs_history']},
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
            if metadata.get('langgraph_node') == 'remember_node':
                continue
            if metadata.get('langgraph_node') == 'summarize_conversation':
                continue

            if metadata.get('langgraph_node') == 'tools':
                continue

            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                full_response += message_chunk.content
                yield message_chunk.content


        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )
    except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)