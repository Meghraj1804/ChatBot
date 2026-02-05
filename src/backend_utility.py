from src.config import checkpointer, vector_db_path, embd_model, search_type, k, chunk_overlap, chunk_size
import os
import json
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Optional
from src.exception import CustomException
from src.logger import logging
import sys
import faiss
from langchain_community.docstore import InMemoryDocstore


def retrieve_all_threads():
    try:
        all_threads = set()
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config['configurable']['thread_id'])
        return list(all_threads)
    except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)


def retrieve_thread_docs():
    try:
        os.makedirs(vector_db_path, exist_ok=True)
        return list(os.listdir(vector_db_path))
    except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

def save_faiss_safe(vector_store: FAISS, path: str) -> None:
    os.makedirs(path, exist_ok=True)

    faiss.write_index(vector_store.index, os.path.join(path, "index.faiss"))

    with open(os.path.join(path, "docstore.json"), "w") as f:
        json.dump(vector_store.docstore._dict, f)

    with open(os.path.join(path, "index_to_docstore_id.json"), "w") as f:
        json.dump(vector_store.index_to_docstore_id, f)

def load_faiss_safe(path: str, embd_model) -> FAISS:
    index_path = os.path.join(path, "index.faiss")

    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found")

    index = faiss.read_index(index_path)

    with open(os.path.join(path, "docstore.json")) as f:
        docstore = InMemoryDocstore(json.load(f))

    with open(os.path.join(path, "index_to_docstore_id.json")) as f:
        index_to_docstore_id = json.load(f)

    return FAISS(
        embedding_function=embd_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

def load_docs(thread_id: str):
    """
    Load an existing FAISS vector store for a thread and return a retriever.
    """
    try:
        print(f"loading {thread_id} docs")

        thread_path = os.path.join(vector_db_path, thread_id)

        if not os.path.isdir(thread_path):
            return {thread_id: {"retriever": None, "documents": []}}

        vector_store = load_faiss_safe(thread_path, embd_model)

        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

        documents = [
            d.metadata.get("source")
            for d in vector_store.docstore._dict.values()
        ]

        return {thread_id: {"retriever": retriever, "documents": documents}}

    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """
    Ingest a PDF into a thread-specific FAISS vector store and return a retriever.
    """
    try:
        if not file_bytes:
            raise ValueError("No bytes received for ingestion.")

        thread_path = os.path.join(vector_db_path, thread_id)
        os.makedirs(thread_path, exist_ok=True)

        vector_store = None

        try:
            vector_store = load_faiss_safe(thread_path, embd_model)
        except FileNotFoundError:
            pass  # first ingest for this thread

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            for d in docs:
                d.metadata.update({
                    "thread_id": thread_id,
                    "source": filename,
                })

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            chunks = splitter.split_documents(docs)

            if vector_store is None:
                vector_store = FAISS.from_documents(chunks, embd_model)
            else:
                vector_store.add_documents(chunks)

            save_faiss_safe(vector_store, thread_path)

            retriever = vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k},
            )

            documents = [
                d.metadata.get("source")
                for d in vector_store.docstore._dict.values()
            ]

            return {
                thread_id: {
                    "retriever": retriever,
                    "documents": documents,
                }
            }

        finally:
            os.remove(temp_path)

    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)




# def load_docs(thread_id: str):
#     """
#     Load an existing FAISS vector store for a thread and return a retriever.
#     """

#     try:
#         print(f'loading {thread_id} docs')

#         thread_path = os.path.join(vector_db_path, thread_id)
        
#         if not os.path.exists(thread_path):
#             return {thread_id: {'retriever': None, 'documents':[]}}

        
#         vector_store = FAISS.load_local(
#             thread_path,
#             embd_model,
#             allow_dangerous_deserialization=True
#         )

#         retriever = vector_store.as_retriever(
#             search_type=search_type,
#             search_kwargs={"k": k}
#         )

#         documents = [i.metadata['source'] for i in list(vector_store.docstore._dict.values())]

#         return {thread_id: {'retriever': retriever, 'documents':documents}}

#     except Exception as e:
#             logging.error(e)
#             raise CustomException(e,sys)


# def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
#     """
#     Ingest a PDF into a thread-specific FAISS vector store and return a retriever.
#     """

#     try:

#         if not file_bytes:
#             raise ValueError("No bytes received for ingestion.")

#         thread_path = os.path.join(vector_db_path, thread_id)
#         os.makedirs(thread_path, exist_ok=True)

#         vector_store = None
#         documents = []


#         if os.path.exists(thread_path):
#             try:
#                 vector_store = FAISS.load_local(
#                     thread_path,
#                     embd_model,
#                     allow_dangerous_deserialization=True
#                 )
#                 documents = [
#                     i.metadata.get("source")
#                     for i in vector_store.docstore._dict.values()
#                 ]
#             except :
#                 vector_store = None


#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file_bytes)
#             temp_path = temp_file.name

#         try:
#             loader = PyPDFLoader(temp_path)
#             docs = loader.load()

#             for d in docs:
#                 d.metadata.update({
#                     "thread_id": thread_id,
#                     "source": filename,
#                 })

#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=chunk_size,
#                 chunk_overlap=chunk_overlap,
#             )
#             chunks = splitter.split_documents(docs)

#             if vector_store is None:
#                 vector_store = FAISS.from_documents(chunks, embd_model)
#             else:
#                 vector_store.add_documents(chunks)

#             vector_store.save_local(thread_path)

#             retriever = vector_store.as_retriever(
#                 search_type=search_type,
#                 search_kwargs={"k": k}
#             )

#             documents = [
#                 i.metadata.get("source")
#                 for i in vector_store.docstore._dict.values()
#             ]

#             return {
#                 thread_id: {
#                     "retriever": retriever,
#                     "documents": documents
#                 }
#             }

#         finally:
#             try:
#                 os.remove(temp_path)
#             except Exception as e:
#                 logging.error(e)
#                 raise CustomException(e,sys)

#     except Exception as e:
#             logging.error(e)
#             raise CustomException(e,sys)
