import os
from dotenv import load_dotenv
from huggingface_hub import login
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["Staging"]  # Select database
collection = db["Users"]  # Select collection

# Define the filter criteria
filter_criteria = {
    "role": "instructor",
    "instructor_request": True,
    "instructor_request_confirmed": True,
    "approve": True
}

# Fetch filtered data from MongoDB
documents = []
for doc in collection.find(filter_criteria, {"_id": 0}):  
    text = " ".join(f"{k}: {v}" for k, v in doc.items())  
    documents.append(Document(text=text))

# Print the documents fetched from MongoDB
print("Fetched documents from MongoDB:")
for document in documents:
    print(document.text)

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set LLM model
Settings.llm = Ollama(model="tinyllama:1.1b", request_timeout=360.0)

if documents:
    # Create an index from MongoDB documents
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    print("\nChatbot is ready! Ask me anything about the approved instructors.\n")
    
    while True:
        query = input("You: ")
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        response = query_engine.query(query)
        print("Chatbot:", response)

else:
    print("No matching instructors found in the database.")
