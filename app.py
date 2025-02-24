import streamlit as st
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

# Load API Key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Streamlit UI
st.set_page_config(page_title="Web Scraper RAG", page_icon="🤗", layout="wide")
st.title("Text Scraping RAG System")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat Interface")
    chat_container = st.container()
    query = st.text_input("Enter your query:", key="chat_input", placeholder="Type a message...")

with col2:
    st.sidebar.subheader("Chat History")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for chat in st.session_state["chat_history"]:
        st.sidebar.write(chat)

def is_valid_url(url):
    """Check if the URL is valid."""
    return validators.url(url)

def scrape_and_process(url):
    """Scrapes data from the website, processes it, and indexes it in FAISS."""
    if not is_valid_url(url):
        st.warning("🚨 ENTER PROPER URL")
        return None, None

    loader = WebBaseLoader(web_paths=(url,))
    
    try:
        docs = loader.load()
    except Exception:
        st.error("❌ Error loading URL. Please check and try again.")
        return None, None
    
    if not docs:
        st.error("😣 No data retrieved from the URL. Try another website.")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    # Create FAISS vector store
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local("faiss_index")

    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(texts)

    st.session_state["vector_db"] = vector_db
    st.session_state["bm25_retriever"] = bm25_retriever
    st.success("🤩 Data successfully scraped and indexed!")

    return vector_db, bm25_retriever

def get_rag_response(query):
    """Retrieves relevant documents and generates a response using LLM if needed."""
    if "vector_db" not in st.session_state or "bm25_retriever" not in st.session_state:
        st.error("👻 No retrievers found. Please scrape data first.")
        return "Error: No retrievers found."

    vector_db = st.session_state["vector_db"]
    bm25_retriever = st.session_state["bm25_retriever"]

    retriever = vector_db.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs:
        st.warning("👻 No exact matches found in FAISS. Trying keyword-based retrieval...")
        retrieved_docs = bm25_retriever.get_relevant_documents(query)

    if retrieved_docs:
        llm = ChatGroq(model_name="Gemma2-9b-It")
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        response = qa_chain.run(query)
    else:
        llm = ChatGroq(model_name="Gemma2-9b-It")
        response = llm.invoke(query)

    st.session_state["chat_history"].append(f"**Query:** {query}\n**Response:** {response}")
    return response

memory = MemorySaver()
workflow = Graph()
workflow.add_node("scraper", scrape_and_process)
workflow.add_node("retriever", get_rag_response)
workflow.set_entry_point("scraper")
workflow.add_edge("scraper", "retriever")
app = workflow.compile(checkpointer=memory)

url = st.sidebar.text_input("Enter website URL:")
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
            chat_container.write(f"**You:** {query}")
            chat_container.write(f"**Bot:** {response}")
    else:
        st.error("👻 No indexed data found. Scrape a website first.")

st.sidebar.write("🫣 Built by [Kirubakaran](https://github.com/kiruba11k)")
