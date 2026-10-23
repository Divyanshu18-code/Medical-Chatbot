from flask import Flask, render_template, request
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
from dotenv import load_dotenv
import os

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
print("[INFO] Loading HuggingFace embedding model...")
embeddings = download_huggingface_embeddings()
print("[INFO] Embedding model loaded successfully.")

# Connect to existing Pinecone index
INDEX_NAME = "medical-chatbot"

print(f"[INFO] Connecting to Pinecone index '{INDEX_NAME}'...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
print("[INFO] Connected to Pinecone index.")

# Create retriever - fetches top 3 most relevant chunks
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize Google Gemini LLM
# Note: models/gemini-3.5-flash is the active, high-speed flash model
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
print("[INFO] Medical RAG Chain initialized successfully.")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form.get("msg") or (request.get_json(silent=True) or {}).get("msg")
    else:
        msg = request.args.get("msg")

    if not msg or not str(msg).strip():
        return "Please enter a valid medical question."

    msg = str(msg).strip()
    print(f"User Input: {msg}")

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

        print(f"Response: {answer[:120]}...")
        return answer
    except Exception as e:
        print(f"Error processing request: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    app.run(host=host, port=port, debug=True)
