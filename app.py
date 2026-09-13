import streamlit as st
import os
import tempfile
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llm_manager import get_llm, get_rag_prompt_template
from rag_pipeline import process_file_and_get_retriever, load_existing_retriever
from evaluation import setup_evaluation

# Set page config
st.set_page_config(page_title="GenAI chatbot", page_icon="🤖")

st.title("🤖 GenAI chatbot")

# Setup Phoenix
if "phoenix_url" not in st.session_state:
    with st.spinner("Initializing Arize Phoenix for observability..."):
        st.session_state.phoenix_url = setup_evaluation()

st.sidebar.markdown(f"[📊 View Arize Phoenix Dashboard]({st.session_state.phoenix_url})")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = load_existing_retriever()
    if st.session_state.retriever:
         st.sidebar.success("Loaded existing document vector store.")

# Sidebar for file upload
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None and "file_processed" not in st.session_state:
    with st.spinner("Processing document and building vector store..."):
        # Save uploaded file to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process file
        try:
             st.session_state.retriever = process_file_and_get_retriever(temp_path)
             st.session_state.file_processed = True
             st.sidebar.success("Document processed successfully!")
        except Exception as e:
             st.sidebar.error(f"Error processing file: {e}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.retriever is None:
            st.warning("Please upload a document first to provide context.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Initialize LLM and Prompt
                    llm = get_llm()
                    rag_prompt = get_rag_prompt_template()
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                        
                    # Execute Retrieval manually to show context later
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    
                    # Create LCEL Chain
                    rag_chain = (
                        {"context": lambda x: format_docs(retrieved_docs), "input": RunnablePassthrough()}
                        | rag_prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Execute Chain
                    answer = rag_chain.invoke(prompt)
                    
                    st.markdown(answer)
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Expandable context section removed per user request
                except Exception as e:
                    st.error(f"An error occurred: {e}")
