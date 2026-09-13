import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_document(file_path):
    """
    Load a document from the given file path based on its extension.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return loader.load()

def chunk_document(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(splits, persist_directory="./chroma_db"):
    """
    Create and persist a Chroma vector store from document splits using HuggingFace embeddings.
    """
    # Using an open-source, fast, and relatively small embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def get_vector_store(persist_directory="./chroma_db"):
    """
    Load an existing vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if directory exists and has files
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
    return None

def process_file_and_get_retriever(file_path):
    """
    Orchestrates loading, chunking, and vector store creation, returning a retriever.
    """
    documents = load_document(file_path)
    splits = chunk_document(documents)
    vectorstore = create_vector_store(splits)
    # Using similarity search retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

def load_existing_retriever():
    """
    Loads existing retriever if vector db is already populated.
    """
    vectorstore = get_vector_store()
    if vectorstore:
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return None
