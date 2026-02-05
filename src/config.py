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
from dotenv import load_dotenv

# load_dotenv()

# EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
# STOCK_PRICE_API_KEY = os.getenv("STOCK_PRICE_API_KEY")


class DecisionOutput(BaseModel):
    decision : Literal["tool_branch", "chat_branch", "rag_branch"] = Field(description="make decision tool_branch or chat_branch")

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    decision : Literal["chat_branch","tool_branch","rag_branch"]
    summary : str
    context : list
    metadata : list

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

search_type = constants.SEARCH_TYPE
k = constants.SEARCH_KWARGS_K
summarization_length = constants.SUMMARIZATION_LENGTH
chunk_size = constants.CHUNK_SIZE
chunk_overlap = constants.CHUNK_OVERLAP


