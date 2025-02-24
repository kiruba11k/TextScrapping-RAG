# Text Scrapping RAG System

## 📌 Project Overview
This project is a **Text Scraping RAG System** that allows users to scrape text data from a given website, index it using **FAISS** and **BM25**, and then retrieve relevant information using **LLM-powered responses**.

## 🚀 Features
- **Web Scraping**: Extracts text content from a given URL.
- **Text Cleaning**: Removes boilerplate text such as privacy policies and terms.
- **Indexing Methods**:
  - **FAISS (Facebook AI Similarity Search)** for vector-based retrieval.
  - **BM25 (Best Matching 25)** for keyword-based retrieval.
- **Retrieval & Querying**:
  - Searches indexed data using FAISS and BM25.
  - If no relevant results are found, falls back to a general **LLM-based response**.
- **LLM Integration**:
  - Uses `ChatGroq` (Gemma2-9b-It) to generate responses.
  - Supports fallback to direct LLM query when retrieval fails.
- **Memory Checkpointing**: Implements `MemorySaver` to store session data.
- **Graph-based Workflow:** Uses LangGraph to manage workflow execution `dynamically`.

## 📂 Project Structure
```
├── app.py  # Streamlit-based web application
├── faiss_index/  # FAISS vector store directory
├── requirements.txt  # Required dependencies
└── README.md  # Project documentation
```

## 📥 Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/kiruba11k/TextScrapping-RAG.git
   cd TextScrapping-RAG
   ```
2. **Create a virtual environment (Optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up API keys:**
   - Add your `GROQ_API_KEY` in `Streamlit secrets`.
   - Add your `GROQ_API_KEY` in `.env`.---->local

## 🛠️ How It Works
1. **Scraping Data**
   - User inputs a **URL** in the Streamlit interface.
   - The system scrapes the text content using `WebBaseLoader`.
   - Unwanted text is removed using a cleaning function.

2. **Indexing**
   - Text is split into chunks using `RecursiveCharacterTextSplitter`.
   - **FAISS indexing**: Converts text chunks into vector embeddings using `HuggingFaceEmbeddings`.
   - **BM25 indexing**: Uses keyword-based retrieval for redundancy.

3. **Querying**
   - User enters a **query**.
   - **FAISS Retriever** searches for similar embeddings.
   - If FAISS fails, **BM25 Retriever** is used.
   - If both fail, **LLM fallback** generates a response.
     
4.  **Workflow Execution with LangGraph**

- **Graph-based Execution:** Uses LangGraph to manage execution paths dynamically.

- **Memory Management:** Implements MemorySaver to track execution history.

- **Routing Logic:**
  
    - If data is retrieved, route to RAG pipeline.

    - If no relevant data, route directly to LLM.
      
- **Graph Nodes:**
  
    - scraper: Loads and processes web data.
      
    -  retriever: Fetches relevant documents using FAISS and BM25.
      
    - router: Routes to either retrieval-based or LLM-based response generation.



📂

## 📡 Running the Application
To launch the Streamlit app:
```sh
streamlit run app.py
```

## 🔎 Retrieval & RAG Pipeline
- **Step 1**: Scrape and preprocess text data.
- **Step 2**: Index the cleaned text into FAISS and BM25.
- **Step 3**: Accept user queries and search indexed data.
- **Step 4**: Retrieve relevant documents and pass them to `ChatGroq`.
- **Step 5**: Generate a final response using RAG (retrieval-augmented generation).

## 🔑 Why FAISS & BM25?
- **FAISS (Vector Similarity Search)**: Efficient for **semantic similarity-based retrieval**.
- **BM25 (Lexical Retrieval)**: Handles **keyword-based searches**, ensuring better fallback.
- Combining both methods enhances **retrieval accuracy and robustness**.

## 📌 Use Cases
- Information retrieval from websites.
- Enhancing chatbot responses with external web data.
- Automating content extraction and summarization.

## 🌐 Streamlit Demo
🔗 [Try the Live Demo](https://textscrapping-rag.streamlit.app/)  

## 📜 Requirements
- Python 3.8+
- Streamlit
- FAISS
- BM25Retriever
- HuggingFace Embeddings
- OpenAI / Groq API
- LangChain

## 🤝 Contributing
Feel free to submit **issues** or **pull requests** to improve this project.

## 📧 Contact
For questions or collaborations, reach out to **[Kirubakaran](https://github.com/kiruba11k)**.

---
**🔧 Built with ❤️ using FAISS, BM25, and LangChain**

