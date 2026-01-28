from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate

from src.config import ChatState, model, decision_model, model_with_tool, tools, checkpointer



# ----------------------------------- 3.Creating Nodes --------------------------------------------------

def decision_node(state : ChatState):
    messages = state['messages']
    for messages in reversed(messages):
        if isinstance(messages, HumanMessage):

            prompt = ChatPromptTemplate.from_messages(
                [('system', 
                'You are a routing assistant. Current date: January 2026.\n\n'
                'Analyze if the user query requires real-time data from external APIs.\n\n'
                'Available APIs:\n'
                '1. Currency conversion (exchange rates)\n'
                '2. Stock prices (Alpha Vantage)\n'
                '3. Calculator tool which will perform addition, subtraction, multiplication and division\n\n'
                'Respond with:\n'
                '- "yes" if the query needs current exchange rates or stock prices\n'
                '- "no" for greetings, general questions, explanations, or historical info\n\n'
                'Examples:\n'
                '- "Hello" → no\n'
                '- "What is inflation?" → no\n'
                '- "Convert 100 USD to EUR" → yes\n'
                '- "What is Apple stock price now?" → yes'
                ),
                ('user','you can use external api 1.Get the conversion factor (exchange rate) between two currencies. 2.Get the latest stock price for a given company symbol using Alpha Vantage API. Do you want to use those api to answer this message {messages}')]
            )

            decision_chain = prompt | decision_model

            output = decision_chain.invoke({'messages':messages})

            print('decision = ',output)

            decision_message = AIMessage(
                content=output.decision,
                name="decision_router",  # Add a name to identify it
            )

            return {'decision':output.decision}

def check_decision(state : ChatState)->Literal["chat_branch","tool_branch"]:
    sentiment = state["decision"]

    if sentiment == 'yes':
        return 'tool_branch'
    else:
        return 'chat_branch'


def chat_branch(state : ChatState):
    messages = state['messages']
    
    output = model.invoke(messages)

    return {'messages':[output]}

def tool_branch(state : ChatState):
    messages = state['messages']

    output = model_with_tool.invoke(messages)

    return {'messages':[output]}

def final_answer_node(state: ChatState):
    # Extract last tool message
    query = state['messages']
    context = state['tool_message']
    

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly using the provided context."),
        ("user", "Query: {query}\nContext: {context}")
    ])

    chain = prompt | model

    output = chain.invoke({
        "query": query,
        "context":context
    })

    return {"messages": [output]}

tool_node = ToolNode(tools)

# ------------------------------------------ 4.Defining Graph ---------------------------------------------

graph = StateGraph(ChatState)

graph.add_node('decision_node',decision_node)
graph.add_node('chat_branch',chat_branch)
graph.add_node('tool_branch',tool_branch)
graph.add_node('final_answer',final_answer_node)
graph.add_node('tools',tool_node)

graph.add_edge(START, 'decision_node')
graph.add_conditional_edges('decision_node',check_decision,{'tool_branch':'tool_branch', 'chat_branch':'chat_branch'})
graph.add_edge('tool_branch','tools')
graph.add_edge('tools','final_answer')
graph.add_edge("final_answer", END)
graph.add_edge('chat_branch', END)


chatbot = graph.compile(checkpointer=checkpointer)


