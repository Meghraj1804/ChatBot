from src import constants
from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing import TypedDict, Annotated, Literal, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from src.extra_tools import tools
from langgraph.store.sqlite import SqliteStore
from langgraph.store.base import BaseStore
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os


class DecisionOutput(BaseModel):
    decision : Literal["yes", "no"] = Field(description="make decision yes or no")

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    decision : Literal["yes", "no"]
    tool_message : Annotated[list[BaseMessage], add_messages]
    summary : str

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


model = ChatOllama(model = constants.MODEL)
embd_model = OllamaEmbeddings(model = constants.EMBD_MODEL)
decision_model = model.with_structured_output(DecisionOutput)
model_with_tool = model.bind_tools(tools)
memory_extractor_model = model.with_structured_output(MemoryDecision)

db_path = os.path.join(constants.DB_FOLDER,constants.DB)
memory_db_path = os.path.join(constants.DB_FOLDER,constants.LT_MEMORY_DB)
os.makedirs(constants.DB_FOLDER, exist_ok=True)
conn_1 = sqlite3.connect(database = db_path, check_same_thread=False, isolation_level=None)
conn_1.execute("PRAGMA journal_mode=WAL;")
checkpointer = SqliteSaver(conn=conn_1)


conn_2 = sqlite3.connect(database = memory_db_path, check_same_thread=False, isolation_level=None)
memory_store = SqliteStore(conn=conn_2)
conn_2.execute("PRAGMA journal_mode=WAL;")
memory_store.setup()


vector_db_path = os.path.join(constants.VECTOR_DB_FOLDER)


