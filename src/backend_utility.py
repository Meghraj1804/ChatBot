from src.config import checkpointer, vector_db_path, embd_model, ChatState
import os
import json
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Optional


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    
    return list(all_threads)

def retrieve_thread_docs():
    os.makedirs(vector_db_path, exist_ok=True)
    return list(os.listdir(vector_db_path))


def load_docs(thread_id: str):
    """
    Load an existing FAISS vector store for a thread and return a retriever.
    """
    print(f'loading {thread_id} docs')

    thread_path = os.path.join(vector_db_path, thread_id)
    
    if not os.path.exists(thread_path):
        return {thread_id: {'retriever': None, 'documents':[]}}

    try:
        vector_store = FAISS.load_local(
            thread_path,
            embd_model,
            allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

        documents = [i.metadata['source'] for i in list(vector_store.docstore._dict.values())]

        return {thread_id: {'retriever': retriever, 'documents':documents}}

    except Exception as e:
        pass


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """
    Ingest a PDF into a thread-specific FAISS vector store and return a retriever.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    thread_path = os.path.join(vector_db_path, thread_id)
    os.makedirs(thread_path, exist_ok=True)

    vector_store = None
    documents = []

    # Try loading existing vector store
    if os.path.exists(thread_path):
        try:
            vector_store = FAISS.load_local(
                thread_path,
                embd_model,
                allow_dangerous_deserialization=True
            )
            documents = [
                i.metadata.get("source")
                for i in vector_store.docstore._dict.values()
            ]
        except Exception:
            vector_store = None

    # Write PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Attach metadata
        for d in docs:
            d.metadata.update({
                "thread_id": thread_id,
                "source": filename,
            })

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        # Create or update vector store
        if vector_store is None:
            vector_store = FAISS.from_documents(chunks, embd_model)
        else:
            vector_store.add_documents(chunks)

        vector_store.save_local(thread_path)

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

        # Refresh document list
        documents = [
            i.metadata.get("source")
            for i in vector_store.docstore._dict.values()
        ]

        return {
            thread_id: {
                "retriever": retriever,
                "documents": documents
            }
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
