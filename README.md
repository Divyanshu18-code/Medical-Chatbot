# 🩺 MedChat AI — End-to-End Medical Chatbot

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-green.svg)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-orange.svg)](https://www.pinecone.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini%203.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, production-grade **Medical Question-Answering Chatbot** powered by **Retrieval-Augmented Generation (RAG)**. The chatbot indexes authoritative medical literature (*The Gale Encyclopedia of Medicine*) into a high-performance **Pinecone Vector Database** and synthesizes verified, medically grounded responses using **Google Gemini** and **LangChain**.

---

## 🌟 Key Features

- **Accurate Medical Information**: Grounded in authentic medical reference literature to minimize hallucinations.
- **Retrieval-Augmented Generation (RAG)**: Retrieves the top relevant textbook passages before generating an answer.
- **Google Gemini 3.5 Flash**: Lightning-fast, cost-effective, and highly coherent response generation.
- **Vector Search with Pinecone**: Scalable, serverless cloud vector store for sub-second semantic retrieval.
- **HuggingFace Embeddings**: Uses `all-MiniLM-L6-v2` (384-dimensional dense vectors) for semantic similarity search.
- **Modern Glassmorphic Web UI**: Dark-mode, responsive chat interface featuring Markdown formatting (bold, bullet points, paragraphs) via Marked.js.
- **Responsible Medical Safeguards**: Prompt-level safety constraints reminding users to seek professional medical advice.

---

## 🏗️ Architecture & How It Works

```
                               ┌────────────────────────┐
                               │   data/Medical_book.pdf│
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │   PyPDFLoader          │ (Extracts 634 pages)
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ RecursiveTextSplitter  │ (chunk_size=500, overlap=50)
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ HuggingFace Embeddings │ (model: all-MiniLM-L6-v2)
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Pinecone Vector Store  │ (Index: medical-chatbot)
                               └───────────┬────────────┘
                                           │
===========================================│===========================================
                                  RUNTIME RAG PIPELINE
===========================================│===========================================
                                           │
    User Query                             │ Top-k relevant chunks
 ("What is asthma?")                       ▼
         │                      ┌────────────────────────┐
         ├─────────────────────►│ Pinecone Retriever     │
         │                      └───────────┬────────────┘
         │                                  │
         │                                  ▼
         │                      ┌────────────────────────┐
         └─────────────────────►│ Medical Prompt Template│ (System + Context + Query)
                                └───────────┬────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Google Gemini 3.5 Flash│
                                └───────────┬────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Flask Web App (/get)   │
                                └───────────┬────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Beautiful Chat UI      │
                                └────────────────────────┘
```

### Detailed Pipeline Breakdown:

1. **Document Loading (`src/helper.py` & `store_index.py`)**:
   - The medical encyclopedia (`Medical_book.pdf`) is loaded using LangChain's `DirectoryLoader` and `PyPDFLoader`.

2. **Chunking & Splitting**:
   - Documents are split into overlapping chunks (500 characters, 50 characters overlap) using `RecursiveCharacterTextSplitter`. Overlap ensures continuity across chunk boundaries.

3. **Embedding Generation**:
   - Each text chunk is converted into a 384-dimensional vector using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`.

4. **Vector Storage (`store_index.py`)**:
   - Embeddings are indexed into a Pinecone Serverless vector database (`medical-chatbot` index) with cosine similarity metric.

5. **Retrieval (`app.py`)**:
   - When a user asks a question, the question is embedded using the same embedding model. Pinecone performs a similarity search to fetch the top 3 most relevant textbook excerpts.

6. **Answer Generation**:
   - A specialized clinical prompt template combines the retrieved excerpts with the user query and feeds them to **Google Gemini 3.5 Flash**. The LLM produces a factual, well-structured answer with clinical disclaimers.

7. **Interactive Web Interface**:
   - The user chats through a responsive web interface built with Flask, HTML5, CSS3, and JavaScript, featuring real-time markdown parsing.

---

## 📁 Project Structure

```
Medical-Chatbot/
│
├── .env.example              # Template for API keys
├── .gitignore                # Git ignore rules (ignores .env and virtual environments)
├── LICENSE                   # Project license (MIT)
├── README.md                 # Project documentation and guide
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup configuration
│
├── app.py                    # Flask web application & RAG query endpoint
├── store_index.py            # One-time script to chunk PDF and upload vectors to Pinecone
│
├── data/
│   └── Medical_book.pdf      # Medical reference textbook
│
├── research/
│   └── trials.ipynb          # Jupyter notebook for experiments and testing
│
├── src/
│   ├── __init__.py           # Package initialization
│   ├── helper.py             # PDF loading, text splitting, & embedding functions
│   └── prompt.py             # System prompt template for medical safety
│
└── templates/
    └── chat.html             # Glassmorphic dark-mode web chat interface
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.10+ | Core development |
| **Framework** | Flask 3.1.1 | Lightweight web server & API |
| **Orchestration** | LangChain / LangChain Classic | RAG chain composition & document processing |
| **LLM** | Google Gemini 3.5 Flash | High-speed, context-aware reasoning |
| **Vector DB** | Pinecone (Serverless) | Fast semantic vector index & similarity search |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | 384-dimensional dense semantic representations |
| **PDF Parser** | PyPDF | Medical book text extraction |
| **Frontend** | HTML5, Vanilla CSS3, JS, Marked.js | Glassmorphism UI with Markdown rendering |

---

## 🚀 Getting Started

Follow these step-by-step instructions to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Divyanshu18-code/Medical-Chatbot.git
cd Medical-Chatbot
```

*(Or navigate into your project directory if already cloned).*

### 2. Create and Activate a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory (you can copy `.env.example`):

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
# Pinecone API Key: https://app.pinecone.io/
PINECONE_API_KEY="your_pinecone_api_key_here"

# Google Gemini API Key: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY="your_google_gemini_api_key_here"
```

> **How to get free API keys:**
> - **Pinecone**: Sign up at [pinecone.io](https://app.pinecone.io/), create a free tier account, and copy your API key from the API Keys tab.
> - **Google Gemini**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and click **Get API Key** (free tier available).

---

### 5. Build the Vector Index (Run Once)

Place your medical PDF inside the `data/` folder (default is `data/Medical_book.pdf`).

Run the index builder to process the PDF and populate your Pinecone database:

```bash
python store_index.py
```

*What this script does:*
1. Loads the PDF from `data/`.
2. Splits the text into ~100k chunks.
3. Generates vector embeddings using `all-MiniLM-L6-v2`.
4. Creates a `medical-chatbot` index in Pinecone if it doesn't already exist.
5. Uploads vectors to Pinecone in batches.

*(Note: If your Pinecone index is already populated with the medical vectors, you can skip this step).*

---

### 6. Run the Medical Chatbot App

Start the Flask server:

```bash
python app.py
```

You will see:
```text
[INFO] Loading HuggingFace embedding model...
[INFO] Embedding model loaded successfully.
[INFO] Connecting to Pinecone index 'medical-chatbot'...
[INFO] Connected to Pinecone index.
[INFO] Medical RAG Chain initialized successfully.
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:8080
```

Open your browser and navigate to:
👉 **[http://localhost:8080](http://localhost:8080)**

---

---

### 7. Production Deployment (Docker & WSGI)

For production environments, run the application using Docker or Gunicorn:

#### Option A: Docker Container (Recommended for Cloud/AWS)
```bash
# Build the Docker image
docker build -t medical-chatbot:latest .

# Run the container (pass your API keys or load .env)
docker run -d -p 8080:8080 --env-file .env --name medchat medical-chatbot:latest

# Health check
curl http://localhost:8080/health
```

#### Option B: Gunicorn WSGI Server (Linux/macOS)
```bash
gunicorn app:app --bind 0.0.0.0:8080 --workers 2 --timeout 120
```

#### Option C: Waitress WSGI Server (Windows Production)
```bash
waitress-serve --port=8080 app:app
```

---

## ⌨️ Keyboard Shortcuts & Features

| Shortcut | Action |
| :--- | :--- |
| `Ctrl + B` (or `Cmd + B`) | Toggle / Collapse Sidebar |
| `Ctrl + K` (or `Cmd + K`) | Start New Chat session |
| `Enter` | Send message |
| `Shift + Enter` | Multiline / newline in input |
| `Esc` | Close mobile sidebar drawer |

---

## 💬 Sample Questions to Ask

- *"What is hypertension and how does it affect the heart?"*
- *"What are the symptoms and causes of diabetes?"*
- *"How does asthma work and what triggers an attack?"*
- *"What is an allergy and how does the immune system react?"*
- *"What are common causes of tension headaches?"*

---

## 🔧 Bugs Fixed & Improvements Made

1. **Resolved Model Deprecation (`404 NOT_FOUND`)**:
   - The original code pointed to `gemini-2.0-flash` which is discontinued. Upgraded to `gemini-3.5-flash`, which is active, fast, and fully supported.

2. **Fixed Duplicate Hamburger Icon & Desktop Sidebar Collapse**:
   - Resolved the duplicate hamburger icon issue (`≡ ≡`) by hiding the navbar toggle button when the sidebar is open on desktop.
   - Implemented smooth desktop collapse animation (`Ctrl+B`) and offcanvas drawer with backdrop for mobile screens.

3. **Production Backend Architecture**:
   - Replaced raw print statements with standard Python `logging`.
   - Added container health check endpoint `/health` returning service status, Pinecone index, and LLM model info.
   - Added HTTP error handlers (`400`, `404`, `500`) and safe input validation.
   - Added `Dockerfile`, `.dockerignore`, and `Procfile` for container and cloud deployments.

4. **Fixed LangChain Chain Import Incompatibilities**:
   - Implemented a flexible import fallback between `langchain.chains` and `langchain_classic.chains`, ensuring stability across modern LangChain versions (0.3+ / 1.x).

5. **Safe Environment Variable Handling**:
   - Added explicit validation for `PINECONE_API_KEY` and `GOOGLE_API_KEY` with clear, informative error messages instead of unhandled runtime crashes.

6. **Multi-Format Response Extraction**:
   - Updated answer extraction logic in `app.py` to seamlessly handle string, `TextAccessor`, and structured list tokens returned by modern LLM interfaces without exposing raw Python dictionaries or signatures.

7. **Batch Uploading in `store_index.py`**:
   - Updated vector uploading to process in batches of 100 chunks, preventing payload limits (413) and network timeout issues with large books.
   - Replaced fixed sleep times with dynamic readiness polling (`pc.describe_index`).

8. **Enhanced ChatGPT UI with Markdown & Speech**:
   - Top `+ New chat` button with local storage multi-chat persistence.
   - Markdown parsing with `Marked.js`, voice dictation (Web Speech API), and text-to-speech audio playback.
   - Glowing clinical AI logo and tailored dark/light aesthetic.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: MedChat AI is an educational demonstration built on Retrieval-Augmented Generation from public medical textbooks. It is **not** a licensed medical practitioner and **does not** provide medical diagnoses, treatment plans, or emergency advice. Always consult a qualified physician or healthcare provider for any medical conditions or health emergencies.

---

## 👨‍💻 Author & Acknowledgements

- **Developer**: Divyanshu Pandey
- **Email**: [divyanshupandey292007@gmail.com](mailto:divyanshupandey292007@gmail.com)
- **GitHub**: [@Divyanshu18-code](https://github.com/Divyanshu18-code)
- **Knowledge Base**: *The Gale Encyclopedia of Medicine*
