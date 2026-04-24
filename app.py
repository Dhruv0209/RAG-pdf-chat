import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# --- CONFIG ---
st.set_page_config(page_title="Hybrid RAG Chat", layout="wide")
st.title("📄 Hybrid RAG Chat with PDFs")

# Cache embeddings to avoid reloading them on every rerun
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# --- CACHED RETRIEVER SETUP ---
@st.cache_resource(show_spinner="Processing Documents...")
def setup_hybrid_retriever(uploaded_files):
    documents = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue()) # Use getvalue() for uploaded files
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = file.name
            documents.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = splitter.split_documents(documents)

    # Vector Store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Keyword Store
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    # Ensemble
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

# --- UI LOGIC ---
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)
    session_id = st.sidebar.text_input("Session ID", value="default")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        retriever = setup_hybrid_retriever(uploaded_files)

        # Chain Construction (moved outside chat input for clarity)
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite query to be standalone based on history. Do NOT answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer ONLY from context. If unknown, say 'I don't know'.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
        )

        # --- CHAT INTERFACE ---
        user_input = st.chat_input("Ask a question...")
        if user_input:
            st.chat_message("user").write(user_input)
            
            with st.chat_message("assistant"):
                # Use INVOKE to get both context and answer in one go to save API costs
                # Then simulate the "streaming" feel if needed, or just display
                with st.spinner("Thinking..."):
                    response = conversational_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.write(response["answer"])

                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Source {i+1}: {doc.metadata.get('source')}**")
                            st.write(doc.page_content[:300] + "...")
                            st.divider()
else:
    st.info("Please provide an API key in the sidebar.")