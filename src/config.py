from src import constants
from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from src.extra_tools import tools
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os


class DecisionOutput(BaseModel):
    decision : Literal["yes", "no"] = Field(description="make decision yes or no")

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    decision : Literal["yes", "no"]
    tool_message : Annotated[list[BaseMessage], add_messages]


model = ChatOllama(model = constants.MODEL)
embd_model = OllamaEmbeddings(model = constants.EMBD_MODEL)
decision_model = model.with_structured_output(DecisionOutput)
model_with_tool = model.bind_tools(tools)

db_path = os.path.join(constants.DB_FOLDER,constants.DB)
os.makedirs(constants.DB_FOLDER, exist_ok=True)
conn = sqlite3.connect(database = db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

vector_db_path = os.path.join(constants.VECTOR_DB_FOLDER)
# os.makedirs(constants.VECTOR_DB_FOLDER, exist_ok=True)