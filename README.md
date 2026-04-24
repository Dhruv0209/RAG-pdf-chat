Hybrid RAG Chat with PDFs 📄🤖

A high-performance RAG (Retrieval-Augmented Generation) application using **Hybrid Search** and **Llama 3.1** via Groq.

## 🚀 Key Features
* **Hybrid Retrieval:** Combines BM25 (keyword search) and ChromaDB (vector search) for 20% better context accuracy.
* **History-Aware:** Remembers previous context to handle follow-up questions.
* **Fast Inference:** Powered by Groq's LPUs for near-instant responses.

## 🛠️ Tech Stack
- **LLM:** Meta Llama 3.1 8B (via Groq)
- **Framework:** LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Frontend:** Streamlit

## 🏁 Getting Started
1. Clone the repo: `git clone https://github.com/yourusername/your-repo-name.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your `GROQ_API_KEY`.
4. Run the app: `streamlit run app.py`
5. Push to GitHub
Open your terminal in the project folder and run these commands:

Bash
# Initialize the repository
git init

# Add all files (the .gitignore will skip your private keys)
git add .

# Commit your changes
git commit -m "Initial commit: Hybrid RAG with Streamlit and Groq"

# Create a new repo on GitHub.com, then link it:
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push to the main branch
git branch -M main
git push -u origin main
🌟 The "Ops" Extra Credit
If you want to show off your MLOps skills, add a Dockerfile. This proves you understand how to containerize AI applications.

Create a file named Dockerfile:

Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
Once this is pushed, your repository will look like a production-ready tool! Do you have a GitHub account set up already, or do you need help with the SSH keys?
