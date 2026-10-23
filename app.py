import logging
import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from src.helper import download_huggingface_embeddings
from src.prompt import system_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("medchat")

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please add it to your .env file.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

app = Flask(__name__)

# Initialize embeddings model
logger.info("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
embeddings = download_huggingface_embeddings()
logger.info("Embedding model loaded successfully.")

# Connect to existing Pinecone index
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")

logger.info(f"Connecting to Pinecone index '{INDEX_NAME}'...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
logger.info("Connected to Pinecone index.")

# Create retriever - fetches top 3 most relevant chunks
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize Google Gemini LLM
# Note: gemini-3.5-flash is the active, high-speed model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
    max_output_tokens=1024,
)

# Build the RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
logger.info("Medical RAG Chain initialized successfully.")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint for load balancers and container orchestrators."""
    return jsonify({
        "status": "healthy",
        "service": "medical-chatbot",
        "version": "1.0.0",
        "index": INDEX_NAME,
        "model": "gemini-3.5-flash"
    }), 200


@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form.get("msg") or (request.get_json(silent=True) or {}).get("msg")
    else:
        msg = request.args.get("msg")

    if not msg or not str(msg).strip():
        return "Please enter a valid medical question.", 400

    msg = str(msg).strip()
    if len(msg) > 3000:
        return "Query is too long. Please limit your message to 3000 characters.", 400

    logger.info(f"User Query: {msg[:80]}...")

    try:
        result = rag_chain.invoke({"input": msg})
        raw_answer = result.get("answer", "")

        # Extract string cleanly across LangChain versions (TextAccessor, list of dicts, or str)
        if hasattr(raw_answer, "text"):
            answer = raw_answer.text
        elif isinstance(raw_answer, list):
            parts = [
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in raw_answer
            ]
            answer = "".join(parts)
        else:
            answer = str(raw_answer)

        logger.info(f"Answer generated ({len(answer)} chars)")
        return answer
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return "I'm sorry, I encountered an error processing your request. Please try again.", 500


@app.errorhandler(400)
def bad_request(e):
    if request.path.startswith("/get") or request.is_json:
        return jsonify({"error": "Bad request", "message": str(e)}), 400
    return render_template("chat.html"), 400


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/get") or request.is_json:
        return jsonify({"error": "Not found"}), 404
    return render_template("chat.html"), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}", exc_info=True)
    if request.path.startswith("/get") or request.is_json:
        return jsonify({"error": "Internal server error"}), 500
    return "An internal server error occurred. Please try again later.", 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    logger.info(f"Starting server on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)

