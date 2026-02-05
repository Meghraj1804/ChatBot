from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
import uuid

from src.propmpt_templates import memory_prompt, decision_prompt, chat_prompt_template, rag_prompt_template
from src.config import ChatState, model, decision_model, model_with_tool, tools, checkpointer,  memory_extractor_model, memory_store



# ----------------------------------- 1.Nodes --------------------------------------------------

def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")
    
    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    last_text = state["messages"][-1].content
    print('remember_node = ',last_text)

    decision = memory_extractor_model.invoke(
                            [
                                SystemMessage(content=memory_prompt.format(user_details_content=existing)),
                                {"role": "user", "content": last_text},
                            ]
                            )
    
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}

def summarize_conversation(state: ChatState):

    existing_summary = state.get("summary", "")

    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
        prompt = "Summarize the conversation above."

    messages_for_summary = state["messages"] + [
        HumanMessage(content=prompt)
    ]
    print('summarize_conversation = ',messages_for_summary)
    response = model.invoke(messages_for_summary)

    # Keep only last 2 messages verbatim
    messages_to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
    }



def decision_node(state : ChatState, config: RunnableConfig, *, store: BaseStore):
    messages = state['messages']
    docs = config["configurable"]["docs"]    
    if len(docs)>0:
        # documents = docs[current_thread_id]['documents']
        return {'decision':'get_context'}
    else:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):


                decision_chain = decision_prompt | decision_model

                print('decision_node = ',message.content)
                
                output = decision_chain.invoke({'messages':message.content})
                print('decision = ',output.decision)

                return {'decision':output.decision}


def chat_branch(state : ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")
    items = store.search(ns)
    user_details = "\n".join(it.value.get("data", "") for it in items) if items else ""

    system_msg = SystemMessage(
        content=chat_prompt_template.format(user_details_content=user_details or "(empty)")
    )

    messages = state['messages']

    output = model.invoke([system_msg] + messages)

    return {'messages':[output]}

def tool_branch(state : ChatState):
    messages = state['messages']

    output = model_with_tool.invoke(messages)

    return {'messages':[output]}

tool_node = ToolNode(tools)

def get_context(state : ChatState, config: RunnableConfig, *, store: BaseStore):
    print('config = ',config)
    current_thread_id = str(config["configurable"]["thread_id"])
    retriever = config["configurable"]["docs"][current_thread_id]['retriever']
    messages = state['messages']

    result = retriever.invoke(messages[-1].content)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {'context':context, 'metadata':metadata}

def rag_branch(state : ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")
    items = store.search(ns)
    user_details = "\n".join(it.value.get("data", "") for it in items) if items else ""

    system_msg = SystemMessage(
        content=chat_prompt_template.format(user_details_content=user_details or "(empty)")
    )

    context = state['context']
    metadata = state['metadata']
    messages = state['messages']

    human_msg = HumanMessage(
        content=rag_prompt_template.format(context=context or "(empty)" , user_input=messages, metadata=metadata or "(empty)")
    )

    output = model.invoke([system_msg] + [human_msg])

    return {'messages':[output]}


# ------------------------------------------- 2.Conditional Nodes ----------------------------------------------------------------

def check_decision(state : ChatState)->Literal["chat_branch","tool_branch","get_context"]:
    sentiment = state["decision"]
    print('sentiment = ',sentiment)
    if sentiment == 'tool_branch':
        return 'tool_branch'
    elif sentiment == 'get_context':
        return 'get_context'
    else:
        return 'chat_branch'

def should_summarize(state: ChatState):
    return len(state["messages"]) > 6

# ------------------------------------------ 3.Defining Graph ---------------------------------------------

graph = StateGraph(ChatState)

graph.add_node("remember_node", remember_node)
graph.add_node('summarize_conversation',summarize_conversation)
graph.add_node('decision_node',decision_node)
graph.add_node('chat_branch',chat_branch)
graph.add_node('tool_branch',tool_branch)
graph.add_node('tools',tool_node)
graph.add_node('get_context',get_context)
graph.add_node('rag_branch',rag_branch)

graph.add_edge(START,'remember_node')
graph.add_conditional_edges('remember_node',should_summarize,{True: "summarize_conversation",False: "decision_node",})
graph.add_edge('summarize_conversation','decision_node')
graph.add_conditional_edges('decision_node',check_decision,{'tool_branch':'tool_branch', 'chat_branch':'chat_branch', 'get_context':'get_context'})
graph.add_conditional_edges('tool_branch',tools_condition,{"tools": "tools", END: END})
graph.add_edge('tools', 'tool_branch')  # After tools, back to agent for next decision
graph.add_edge('get_context','rag_branch')
graph.add_edge('rag_branch',END)
graph.add_edge('chat_branch', END)


chatbot = graph.compile(checkpointer=checkpointer, store=memory_store)


