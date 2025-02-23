import streamlit as st
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
st.set_page_config(page_title="Web Scraper RAG", page_icon="🌍", layout="wide")
st.title(" Text Scraping RAG System")

url = st.sidebar.text_input("Enter website URL:")
query = st.text_input("Enter your query:")

def clean_text(text):
    """Removes unwanted boilerplate content from scraped data."""
    unwanted_phrases = ["Privacy Policy", "Terms of Service", "Subscribe", "Contact Us", "Copyright"]
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")
    return text.strip()

def scrape_and_process(url):
    """Scrapes data from the website, processes it, and indexes it in FAISS."""
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    
    if not docs:
        st.error("❌ No data retrieved from the URL. Try another website.")
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
    st.success("✅ Data successfully scraped and indexed!")

    return vector_db, bm25_retriever

def get_rag_response(query):
    """Retrieves relevant documents and generates a response using LLM."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Ensure retrievers are available
    if "vector_db" not in st.session_state or "bm25_retriever" not in st.session_state:
        st.error("⚠️ No retrievers found. Please scrape data first.")
        return "Error: No retrievers found."

    vector_db = st.session_state["vector_db"]
    bm25_retriever = st.session_state["bm25_retriever"]

    # First try FAISS retrieval
    retriever = vector_db.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)

    # If FAISS retrieval is empty, fall back to BM25 retrieval
    if not retrieved_docs:
        st.warning("⚠️ No exact matches found in FAISS. Trying keyword-based retrieval...")
        retrieved_docs = bm25_retriever.get_relevant_documents(query)

    # If still no results, use general LLM without retrieval
    if not retrieved_docs:
        st.warning("⚠️ No relevant data found. Using general LLM response...")
        llm = ChatGroq(model_name="Gemma2-9b-It")
        return llm.invoke(query)  # Direct LLM response

    # Use the best retrieval method found and generate response
    llm = ChatGroq(model_name="Gemma2-9b-It")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.run(query)

    return response

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
        st.error("❌ Please enter a valid URL.")

if query:
    if "vector_db" in st.session_state:
        with st.spinner("🔍 Searching relevant information..."):
            response = get_rag_response(query)
            st.write("**Query:**", query)
            st.write("**Result:**", response)
    else:
        st.error("⚠️ No indexed data found. Scrape a website first.")

st.sidebar.write("⚡ Built by [Kirubakaran](https://github.com/kiruba11k)")
