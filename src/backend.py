from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
import uuid

from src.propmpt_templates import memory_prompt, decision_prompt, chat_prompt_template
from src.config import ChatState, model, decision_model, model_with_tool, tools, checkpointer,  memory_extractor_model, memory_store



# ----------------------------------- 1.Nodes --------------------------------------------------

def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    last_text = state["messages"][-1].content

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

    response = model.invoke(messages_for_summary)

    # Keep only last 2 messages verbatim
    messages_to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
    }



def decision_node(state : ChatState):
    messages = state['messages']
    for message in reversed(messages):
        if isinstance(message, HumanMessage):

            decision_chain = decision_prompt | decision_model

            output = decision_chain.invoke({'messages':message})

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

def final_answer_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value.get("data", "") for it in items) if items else ""

    system_msg = SystemMessage(
        content=chat_prompt_template.format(user_details_content=user_details or "(empty)")
    )
    messages = state['messages']

    output = model.invoke([system_msg] + messages)
    

    return {"messages": [output]}

tool_node = ToolNode(tools)

# ------------------------------------------- 2.Conditional Nodes ----------------------------------------------------------------

def check_decision(state : ChatState)->Literal["chat_branch","tool_branch"]:
    sentiment = state["decision"]

    if sentiment == 'yes':
        return 'tool_branch'
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
graph.add_node('final_answer',final_answer_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'remember_node')
graph.add_conditional_edges('remember_node',should_summarize,{True: "summarize_conversation",False: "decision_node",})
graph.add_edge('summarize_conversation','decision_node')
graph.add_conditional_edges('decision_node',check_decision,{'tool_branch':'tool_branch', 'chat_branch':'chat_branch'})
graph.add_edge('tool_branch','tools')
graph.add_edge('tools','final_answer')
graph.add_edge("final_answer", END)
graph.add_edge('chat_branch', END)


chatbot = graph.compile(checkpointer=checkpointer, store=memory_store)


