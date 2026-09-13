# GenAI Chatbot (Llama 3 + Arize Phoenix)

**Project Demo Video:** [Watch on YouTube](https://youtu.be/lIiJBtr44Dg)

A powerful Generative AI chatbot built with Streamlit, leveraging Retrieval-Augmented Generation (RAG) to provide accurate answers based on user-uploaded documents. This project uses Groq's fast inference API for LLM responses and Arize Phoenix for comprehensive AI observability and evaluation.

## Features
- **Document Processing**: Upload PDF or TXT files directly from the UI.
- **Retrieval-Augmented Generation (RAG)**: Answers questions based strictly on the uploaded context to reduce hallucinations.
- **AI Observability**: Deep integration with Arize Phoenix to trace LLM calls, monitor token usage, and evaluate retrieval performance.
- **Fast Inference**: Powered by Groq's high-speed API.

## Technical Architecture

### 1. Chunking Method
The application uses the `RecursiveCharacterTextSplitter` from LangChain. 
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 200 characters
This method ensures that paragraphs and sentences are kept intact as much as possible while maintaining a context window suitable for the embedding model.

### 2. Embedding Model
We use `HuggingFaceEmbeddings` with the `all-MiniLM-L6-v2` model. This is a highly efficient, open-source sentence-transformers model that provides dense vector representations for semantic search.

### 3. Vector Database
The project utilizes **ChromaDB**, an open-source vector database. It runs locally and persistently stores the document embeddings, allowing for lightning-fast similarity search during the retrieval phase of the RAG pipeline.

### 4. RAG Pipeline Explanation
1. **Ingestion:** The user uploads a document. The file is parsed and split into smaller chunks using the chunking strategy.
2. **Embedding:** Each chunk is passed through the HuggingFace embedding model to generate a high-dimensional vector.
3. **Storage:** The vectors and their corresponding text chunks are stored in ChromaDB.
4. **Retrieval:** When a user asks a question, the query is embedded using the same model. ChromaDB performs a similarity search to find the most relevant chunks.
5. **Generation:** The retrieved chunks are injected into a prompt alongside the user's question. The LLM (running via Groq) generates a natural, accurate response based *only* on the provided context.

## Setup Instructions

### Prerequisites
- Python 3.11+
- A Groq API Key

### Installation
1. Clone this repository:
   ```bash
   git clone <your-github-repo-url>
   cd aiclone
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your Environment Variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Application

1. **Start the Arize Phoenix Evaluation Server:**
   Open a terminal and run the Phoenix server on port 6007:
   ```bash
   $env:PHOENIX_GRPC_PORT="4318"
   $env:PHOENIX_PORT="6007"
   phoenix serve
   ```

2. **Start the Streamlit App:**
   Open a second terminal and run:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501` to use the chatbot, and `http://localhost:6007` to view the Arize Phoenix observability dashboard.
