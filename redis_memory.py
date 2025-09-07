import redis
import json
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

CACHE_TTL = 60 * 60  # 1 hour

def get_cached_answer(question: str, session_id: str = "default"):
    key = f"{session_id}:{question}"
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)  # returns {"answer": ..., "sources": ...}
    return None

def set_cached_answer(question: str, session_id: str, answer: str, sources: list):
    key = f"{session_id}:{question}"
    value = json.dumps({"answer": answer, "sources": sources})
    redis_client.setex(key, CACHE_TTL, value)
