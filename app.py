import streamlit as st
import re
import validators
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.retrievers import BM25Retriever
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import Graph
from langgraph.checkpoint.memory import MemorySaver
import requests

# Load API Key from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = groq_api_key

# Streamlit UI
st.set_page_config(page_title="Web Scraper RAG", page_icon="🤗", layout="wide")
st.title(" Text Scraping RAG System")

url = st.sidebar.text_input("Enter website URL:")
query = st.text_input("Enter your query:")

def is_valid_url(url):
    """Check if the URL is valid."""
    return validators.url(url)

def clean_text(text):
    """Cleans the extracted text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_and_process(url):
    """Scrapes data from the website, processes it, and indexes it in FAISS."""
    if not is_valid_url(url):
        st.warning("🚨 ENTER PROPER URL")
        return None, None

    loader = WebBaseLoader(web_paths=(url,))
    try:
        docs = loader.load()
    except requests.exceptions.MissingSchema:
        st.error("❌ Invalid URL format. Please enter a proper URL.")
        return None, None
    
    if not docs:
        st.error("😣 No data retrieved from the URL. Try another website.")
        return None, None
    
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata = {"source": url}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    # Create FAISS vector store
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local("faiss_index")

    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(texts)

    # Store retrievers in session state
    st.session_state["vector_db"] = vector_db
    st.session_state["bm25_retriever"] = bm25_retriever
    st.success("🤩 Data successfully scraped and indexed!")

    return vector_db, bm25_retriever

def get_rag_response(query):
    """Retrieves relevant documents and generates a response using LLM."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    if "vector_db" not in st.session_state or "bm25_retriever" not in st.session_state:
        st.error("👻 No retrievers found. Please scrape data first.")
        return "Error: No retrievers found."

    vector_db = st.session_state["vector_db"]
    bm25_retriever = st.session_state["bm25_retriever"]

    # FAISS retrieval
    retriever = vector_db.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""

    # BM25 fallback
    if not retrieved_docs:
        st.warning("👻 No exact matches found in FAISS. Trying keyword-based retrieval...")
        retrieved_docs = bm25_retriever.get_relevant_documents(query)
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""

    # LLM response
    llm = ChatGroq(model_name="Gemma2-9b-It")
    llm_response = llm.invoke(query)

    # Display results
    final_response = """**Retrieved Answer (RAG):**
    {}

    **LLM Generated Answer:**
    {}""".format(retrieved_text if retrieved_text else "🚫 No relevant RAG content found.", llm_response)

    return final_response

# Define workflow and memory checkpointing
memory = MemorySaver()
workflow = Graph()
workflow.add_node("scraper", scrape_and_process)
workflow.add_node("retriever", get_rag_response)
workflow.set_entry_point("scraper")
workflow.add_edge("scraper", "retriever")

# Compile workflow with memory saving
app = workflow.compile(checkpointer=memory)

if st.sidebar.button("Scrape & Process"):
    if url:
        with st.spinner("Scraping and indexing data..."):
            vector_db, bm25_retriever = scrape_and_process(url)
            if vector_db:
                st.session_state["vector_db"] = vector_db
                st.session_state["bm25_retriever"] = bm25_retriever
    else:
        st.error("😣 Please enter a valid URL.")

if query:
    if "vector_db" in st.session_state:
        with st.spinner("🧐 Searching relevant information..."):
            response = get_rag_response(query)
            st.write("**Query:**", query)
            st.write("**Result:**")
            st.write(response)
    else:
        st.error("👻 No indexed data found. Scrape a website first.")

st.sidebar.write("🫣 Built by [Kirubakaran](https://github.com/kiruba11k)")
