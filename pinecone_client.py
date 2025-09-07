from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, INDEX_NAME

# Create client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index
index = pc.Index(INDEX_NAME)

# Upsert helper
def upsert_vectors(ids, vectors, metadata):
    items = [{"id": i, "values": v, "metadata": m} for i, v, m in zip(ids, vectors, metadata)]
    index.upsert(vectors=items)

# Query helper
def query_index(vector, top_k=5,include_metadata=True):
    return index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)
