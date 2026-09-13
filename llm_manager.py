import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_llm(model_name="qwen/qwen3.8-27b", temperature=0.0):
    """
    Initialize the Groq LLM client.
    """
    # Ensure GROQ_API_KEY is available in the environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
        
    llm = ChatGroq(
        temperature=temperature,
        model_name=model_name,
        groq_api_key=groq_api_key,
        max_tokens=800
    )
    return llm

def get_rag_prompt_template():
    """
    Define the prompt template for the RAG pipeline.
    """
    system_prompt = (
        "You are Alex, a highly professional, polite, and helpful AI assistant. "
        "Your goal is to answer the user's questions in a natural, conversational, and human-like manner. "
        "Use the following pieces of retrieved context to answer the user's question accurately. "
        "If you don't know the answer, politely inform the user that you don't have that information. "
        "Do not explicitly mention that you are using 'retrieved context' or reading from a document unless asked. "
        "Keep your answers concise, well-structured, and use decent, respectful language at all times. "
        "Never refuse a harmless request abruptly; always maintain a professional tone."
        "\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt
