from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, INDEX_NAME

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# List existing indexes
existing_indexes = pc.list_indexes().names()
# Create index if it doesn't exist
if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{INDEX_NAME}' created.")

# Connect to the index
index = pc.Index(name=INDEX_NAME)

# Helper: Upsert vectors
def upsert_vectors(ids, vectors, metadata):
    items = [{"id": i, "values": v, "metadata": m} for i, v, m in zip(ids, vectors, metadata)]
    index.upsert(vectors=items)

# Helper: Query index
def query_index(vector, top_k=5, include_metadata=True):
    return index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)
