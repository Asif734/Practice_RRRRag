from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

embed_model = SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts):
    return embed_model.encode(texts, show_progress_bar=False).tolist()
