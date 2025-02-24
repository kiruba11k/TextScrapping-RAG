import streamlit as st
import validators
import time
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import Graph
from langgraph.checkpoint.memory import MemorySaver

# Load API Key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Streamlit UI Configuration
st.set_page_config(page_title="Web Scraper RAG", page_icon="🤗", layout="wide")
st.markdown("""
    <style>
    body {background: linear-gradient(to right, #ece9e6, #ffffff);}
    .stChatInput div[role="textbox"] {border-radius: 20px; padding: 10px; font-size: 16px;}
    .stTextInput>div>div>input {border-radius: 10px; padding: 8px; font-size: 16px;}
    .stButton button {border-radius: 10px; padding: 10px 20px; background-color: #ff4b4b; color: white;}
    </style>
""", unsafe_allow_html=True)

# Layout Setup
col1, col2 = st.columns([3, 1])

# Chat History Sidebar
st.sidebar.header("💬 Chat History")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
for chat in st.session_state["chat_history"]:
    st.sidebar.write(chat)

# URL Input Section
with col1:
    url = st.text_input("🌐 Enter website URL:", key="url_input")
    
    def is_valid_url(url):
        return validators.url(url)

    def scrape_and_process(url):
        if not is_valid_url(url):
            st.warning("🚨 Enter a valid URL!")
            return None, None
        loader = WebBaseLoader(web_paths=(url,))
        try:
            docs = loader.load()
        except Exception:
            st.error("❌ Error loading URL. Please check and try again.")
            return None, None
        if not docs:
            st.error("😣 No data retrieved. Try another website.")
            return None, None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        texts = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        vector_db = FAISS.from_documents(texts, embeddings)
        vector_db.save_local("faiss_index")
        bm25_retriever = BM25Retriever.from_documents(texts)
        st.session_state["vector_db"] = vector_db
        st.session_state["bm25_retriever"] = bm25_retriever
        st.success("✅ Data scraped & indexed successfully!")
        return vector_db, bm25_retriever

    if st.button("🚀 Scrape & Process"):
        if url:
            with st.spinner("Scraping & indexing data..."):
                scrape_and_process(url)
        else:
            st.error("😣 Please enter a valid URL.")

# Chat Interface at the Bottom
query = st.chat_input("💬 Enter your query:")

def get_rag_response(query):
    if "vector_db" not in st.session_state or "bm25_retriever" not in st.session_state:
        st.error("👻 No indexed data found. Scrape a website first.")
        return "Error: No retrievers found."
    retriever = st.session_state["vector_db"].as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)
    if not retrieved_docs:
        st.warning("👻 No exact match in FAISS. Trying keyword-based retrieval...")
        retrieved_docs = st.session_state["bm25_retriever"].get_relevant_documents(query)
    if retrieved_docs:
        llm = ChatGroq(model_name="Gemma2-9b-It")
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        response = qa_chain.run(query)
    else:
        llm = ChatGroq(model_name="Gemma2-9b-It")
        response = llm.invoke(query)
    return response

if query:
    if "vector_db" in st.session_state:
        with st.spinner("🤔 Thinking..."):
            response = get_rag_response(query)
            st.session_state["chat_history"].append(f"**You:** {query}")
            st.session_state["chat_history"].append(f"**Bot:** {response}")
            st.write("**You:**", query)
            st.write("**Bot:**")
            for char in response:
                time.sleep(0.03)  # Simulating typing effect
                st.markdown(char, unsafe_allow_html=True)
    else:
        st.error("👻 No indexed data found. Scrape a website first.")

st.sidebar.write("🛠 Built by [Kirubakaran](https://github.com/kiruba11k)")
