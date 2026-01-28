from src.config import checkpointer, vector_db_path, embd_model
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


def load_docs(thread_id:str):
    thread_data = {}

    thread_path = os.path.join(vector_db_path, thread_id)

    if not os.path.exists(thread_path):
        return thread_data  # Nothing to load


    # Load FAISS vector store
    vector_store = FAISS.load_local(
        thread_path,
        embd_model,
        allow_dangerous_deserialization=True
    )

    keys_path = os.path.join(thread_path, "doc_keys.json")

    with open(keys_path, "r") as f:
        doc_keys = json.load(f)

        # Convert to retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": k})

        # Get all documents (for metadata access)
        documents = [vector_store.docstore.search(key) for key in doc_keys]
        thread_data[thread_id] = {
            "retriever": retriever,
            "documents": documents
        }

    return thread_data


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):

    print(f'ingesting {filename}')

    thread_db_path = os.path.join(vector_db_path, thread_id)
    os.makedirs(thread_db_path, exist_ok=True)

    # Load existing vector store if it exists
    vector_store = None
    keys_path = os.path.join(thread_db_path, "doc_keys.json")
    if os.path.exists(thread_db_path):
        try:
            vector_store = FAISS.load_local(
                thread_db_path,
                embd_model,
                allow_dangerous_deserialization=True
            )
        except Exception:
            vector_store = None

    # Load existing keys
    existing_keys = []
    if os.path.exists(keys_path):
        with open(keys_path, "r") as f:
            existing_keys = json.load(f)

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    # Write PDF bytes to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Add metadata and generate keys
        new_keys = []
        start_index = len(existing_keys)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "thread_id": thread_id,
                "file_name": filename
            })
            key = f"{thread_id}_{start_index + i}"
            chunk.metadata["key"] = key
            new_keys.append(key)

        # Add documents to vector store
        if vector_store is None:
            # No existing store, create new
            vector_store = FAISS.from_documents(chunks, embd_model)
        else:
            # Add new documents to existing store
            vector_store.add_documents(chunks)

        # Save updated vector store
        vector_store.save_local(thread_db_path)

        # Update keys JSON
        all_keys = existing_keys + new_keys
        with open(keys_path, "w") as f:
            json.dump(all_keys, f)

        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        return {thread_id: {'retriever': retriever, 'documents': filename}}

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def get_context(user_input, retriever):
    result = retriever.invoke(user_input)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    user_input = f'''
                        You are a helpful assistant answering questions based on retrieved documents.

                        Rules:
                        - Use only the information in the context
                        - Do not speculate or add external knowledge
                        - If the answer is partial, say so clearly
                        - If no answer exists, say you don’t know

                        <context>
                        {context}
                        </context>

                        User asks:
                        {user_input}

                        Metadata:
                        {metadata}

                        Answer in a clear, friendly, and professional tone.

                        '''

    return user_input

    
