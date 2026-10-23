from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf_file(data_path):
    """Load all PDF files from the specified directory."""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_split(extracted_data):
    """Split extracted documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_huggingface_embeddings():
    """Download and return HuggingFace embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return embeddings
