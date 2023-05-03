import os
import sys
import time
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

def load_data():
    # Get a list of all PDF files in localknowledgebase folder
    pdf_files = [f for f in os.listdir("./localknowledgebase") if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in the localknowledgebase folder.")
        sys.exit(1)

    texts = []

    # Load all PDF files found in localknowledgebase folder
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join("./localknowledgebase", pdf_file))
        pages = loader.load_and_split()
        texts.extend(pages)

    return texts

# Check if the index is ready every 300 seconds
def wait_for_index_creation(pinecone_client, index_name):
    while True:
        index_info = pinecone_client.describe_index(index_name)
        index_status = index_info["status"]["state"]
        index_ready = index_info["status"]["ready"]
        if index_status == "Ready" and index_ready:
            break
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Waiting for index creation... (current state: {index_status}, ready: {index_ready})")
        time.sleep(300)

def main():
    # Load environment variables
    load_dotenv()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Environment variables loaded.")

    # Load data
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Loading data...")
    pages = load_data()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Data loaded.")

    # Chunk data up into smaller documents
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Splitting data into smaller documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)

    # Show the total number of documents
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Now you have {len(texts)} documents")

    # Preview first and last documents and ask for user confirmation
    print("\nFirst document preview:")
    print(texts[0].page_content)
    print("\nLast document preview:")
    print(texts[-1].page_content)

    # Ask the user if they want to continue
    user_input = input("\nDo you want to continue (y/n)? ").strip().lower()
    if user_input != 'y':
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Aborting process.")
        sys.exit(0)

    # Create embeddings of documents to get ready for semantic search
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Creating embeddings...")
    api_key = os.environ.get('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Embeddings created.")

    # Initialize Pinecone
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Initializing Pinecone...")
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_api_env = os.environ.get('PINECONE_API_ENV')
    index_name = os.environ.get('INDEX_NAME')
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_api_env)
    pinecone_client = pinecone.Index(index_name)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone initialized.")

    # Check if index already exists
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embeddings.dimension)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Creating index: {index_name}")
        wait_for_index_creation(pinecone_client, index_name)
    else:
        # Update the index with new text, embeddings, and metadata
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Updating index with new texts, embeddings, and metadata...")
        docsearch = Pinecone.from_texts(texts=[t.page_content for t in texts], embedding=embeddings, metadatas=[t.metadata for t in texts], index_name=index_name)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Updated index with {len(texts)} documents")

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Index created/updated and embeddings stored. You can now run query.py to perform semantic search.")

if __name__ == "__main__":
    main()