import streamlit as st
import validators
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.retrievers import BM25Retriever
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import Graph
from langgraph.checkpoint.memory import MemorySaver
import html

# Load API Key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Streamlit UI
st.set_page_config(page_title="Web Scraper RAG", page_icon="🤗", layout="wide")
st.title("Text Scraping RAG System")

# Layout Setup
col1, col2 = st.columns([3, 1])

# Chat history in right sidebar
st.sidebar.header("Chat History")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history in reverse order
for i, chat in enumerate(reversed(st.session_state["chat_history"])):
    query_escaped = html.escape(chat["query"])
    response_escaped = html.escape(chat["response"])

    with st.sidebar.expander(f"📋 {query_escaped[:30]}...", expanded=False):
        st.markdown(f"""
            <div id="chat-{i}" style="border: 1px solid #ddd; padding: 8px; margin-bottom: 5px; 
                        border-radius: 8px; background-color: #f7f7f7; word-wrap: break-word;">
                <strong style="color: #333;">You:</strong> {query_escaped}<br>
                <strong style="color: #007bff;">Bot:</strong> {response_escaped}
            </div>
        """, unsafe_allow_html=True)

with col1:
    url = st.text_input("Enter website URL:", key="url_input")

    def is_valid_url(url):
        return validators.url(url)

    def scrape_and_process(url):
        """Scrapes and processes the website content for indexing."""
        if not is_valid_url(url):
            st.warning("😵 Enter the Proper URL")
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

        # Split text and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        texts = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

        # FAISS Vector Store
        vector_db = FAISS.from_documents(texts, embeddings)
        vector_db.save_local("faiss_index")

        # BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(texts)

        st.session_state["vector_db"] = vector_db
        st.session_state["bm25_retriever"] = bm25_retriever
        st.success("🤩 Data successfully scraped and indexed!")

        return vector_db, bm25_retriever

    def get_rag_response(query):
        """Retrieves relevant documents using FAISS & BM25."""
        if "vector_db" not in st.session_state or "bm25_retriever" not in st.session_state:
            st.error("👻 No retrievers found. Please scrape data first.")
            return []

        vector_db = st.session_state["vector_db"]
        bm25_retriever = st.session_state["bm25_retriever"]

        retriever = vector_db.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            retrieved_docs = bm25_retriever.get_relevant_documents(query)

        return retrieved_docs

    def route_llm(query, retrieved_docs):
        """Routes to the correct LLM if no relevant RAG content is found."""
        groq_llm = ChatGroq(model_name="Gemma2-9b-It")

        if retrieved_docs:
            # Use RAG-based response
            qa_chain = RetrievalQA.from_chain_type(groq_llm, retriever=st.session_state["vector_db"].as_retriever())
            response = qa_chain.run(query)
        else:
            # Use direct LLM if no retrieved content
            response = groq_llm.invoke(query)

        st.session_state.chat_history.append({"query": query, "response": response})
        return response

    def route_based_on_docs(docs):
        """Determines next step based on retrieved documents."""
        return "router" if data and isinstance(data, list) and len(data) > 0 else "llm"

    # Graph Workflow
    memory = MemorySaver()
    workflow = Graph()

    workflow.add_node("scraper", scrape_and_process)
    workflow.add_node("retriever", get_rag_response)
    workflow.add_node("router", route_llm)

    # Workflow
    workflow.set_entry_point("scraper")
    workflow.add_edge("scraper", "retriever")
    workflow.add_conditional_edges("retriever", route_based_on_docs)
    app = workflow.compile(checkpointer=memory)

    # Scrape & Process Button
    if st.button("Scrape & Process"):
        if url:
            with st.spinner("Scraping and indexing data..."):
                vector_db, bm25_retriever = scrape_and_process(url)
                if vector_db:
                    st.session_state["vector_db"] = vector_db
                    st.session_state["bm25_retriever"] = bm25_retriever
        else:
            st.error("😣 Please enter a valid URL.")

# Chat Interface
query = st.chat_input("Enter your query:")

if query:
    if "vector_db" in st.session_state:
        with st.spinner("🧐 Searching relevant information..."):
            retrieved_docs = get_rag_response(query)

            response = route_llm(query, retrieved_docs)

            if not any(chat["query"] == query and chat["response"] == response for chat in st.session_state.chat_history):
                st.session_state.chat_history.append({"query": query, "response": response})

            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                st.write(response)
    else:
        st.error("👻 No indexed data found. Scrape a website first.")

st.sidebar.write("🫣 Built by [Kirubakaran](https://github.com/kiruba11k)")
