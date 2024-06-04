from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import faiss
import pickle

# Load environment variables
load_dotenv()

# Load the Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Function to perform vector embedding
def embed_pdfs():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./constitution")  # Data Ingestion
    docs = loader.load()  # Document Loading
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    final_documents = text_splitter.split_documents(docs[:20])  # Splitting
    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector OpenAI embeddings
    return vectors, final_documents

# Embed PDFs and save vectors
vectors, final_documents = embed_pdfs()
faiss.write_index(vectors.index, "vectors.index")

# Save the documents separately
with open("documents.pkl", "wb") as f:
    pickle.dump(final_documents, f)