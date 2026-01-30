import streamlit as st
from src.backend_utility import retrieve_all_threads, retrieve_thread_docs, ingest_pdf, load_docs, get_context
from src.frontend_utility import reset_chat, load_conversations, ai_only_stream
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_messages' not in st.session_state:
    st.session_state['thread_messages'] = retrieve_all_threads()

if 'current_thread_id'not in st.session_state:
    st.session_state['current_thread_id'] = reset_chat()

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = retrieve_thread_docs()

if 'thread_docs_history' not in st.session_state:
    st.session_state['thread_docs_history'] = {}

if "user_id" not in st.session_state:
    st.session_state["user_id"] = '18'

current_thread_id = str(st.session_state['current_thread_id'])


st.sidebar.markdown(f"**Thread ID:** `{st.session_state['current_thread_id']}`")
if st.sidebar.button('New Chat'):
    reset_chat()

for thread_id in st.session_state['thread_messages']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['current_thread_id'] = thread_id
        messages = load_conversations()
        st.session_state['thread_docs_history'] = load_docs(current_thread_id)
            
        temp_msg = []

        for msg in messages:
            if msg.content == '' or "{" in msg.content and "}" in msg.content:
                continue
            if isinstance(msg, HumanMessage):
                role = 'user'
            if isinstance(msg, AIMessage):
                role = 'assistant'
            temp_msg.append({'role':role, 'content':msg.content})
        st.session_state['message_history'] = temp_msg

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

if uploaded_pdf:
    if uploaded_pdf.name == st.session_state['thread_docs_history'][current_thread_id]['documents']:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
                st.session_state['thread_docs_history'] = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=current_thread_id,
                    filename=uploaded_pdf.name,
                )
                print(st.session_state['thread_docs_history'])
                uploaded_pdf = None

user_input = st.chat_input('Type here')
if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    if len(st.session_state['thread_docs_history']) != 0:
        user_input = get_context(user_input, st.session_state['thread_docs_history'][current_thread_id]['retriever'])
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):

        ai_message = st.write_stream(ai_only_stream(user_input))

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})