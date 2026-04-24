# 📄 Hybrid RAG Chat with PDFs 🤖

A high-performance **Retrieval-Augmented Generation (RAG)** application using **Hybrid Search** and **Llama 3.1 via Groq** for fast and accurate document-based Q&A.

---

## 🚀 Key Features

* **Hybrid Retrieval:** Combines BM25 (keyword search) + ChromaDB (vector search) for improved accuracy
* **Conversational Memory:** Handles follow-up questions using chat history
* **Source Attribution:** Displays document sources for transparency
* **Fast Inference:** Powered by Groq LPUs for near real-time responses
* **Multi-PDF Support:** Query across multiple uploaded documents

---

## 🛠️ Tech Stack

* **LLM:** Meta Llama 3.1 (via Groq)
* **Framework:** LangChain
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Frontend:** Streamlit

---

## 🏁 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-pdf-chat.git
cd rag-pdf-chat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token (optional)
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
rag-pdf-chat/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
```

---

## 💡 Example Use Cases

* Ask questions from research papers
* Summarize PDFs
* Extract key insights from documents
* Conversational querying over knowledge base

---

## ⚠️ Notes

* Do NOT upload `.env` file to GitHub
* Add `.env` in `.gitignore`
* Use a valid Groq API key

---

## ⭐ Future Improvements

* Add streaming responses
* Deploy on Streamlit Cloud / AWS
* Add agent-based workflows
* Implement evaluation metrics

---

## 🧠 About

This project demonstrates a **production-ready GenAI system** using:

* Hybrid Retrieval (semantic + keyword)
* Context-aware conversations
* Efficient LLM usage

---

## 
