import requests
from config import OLLAMA_URL, MODEL_NAME

# Maintain chat history per session
chat_histories = {}

def ask_ollama(prompt: str, memory: bool = True, session_id: str = "default"):
    global chat_histories

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    # Build conversation context
    if memory:
        session_prompt = "\n".join(chat_histories[session_id] + [f"User: {prompt}", "Assistant:"])
    else:
        session_prompt = prompt

    payload = {
        "model": MODEL_NAME,
        "prompt": session_prompt,
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    answer = resp.json().get("response", "")

    # Save conversation to history
    if memory:
        chat_histories[session_id].append(f"User: {prompt}")
        chat_histories[session_id].append(f"Assistant: {answer}")

    return answer
