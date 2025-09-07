import requests
from config import OLLAMA_URL, MODEL_NAME

# Optional: maintain session memory for interactive chat
chat_history = []

def ask_ollama(prompt, memory=True):
    global chat_history
    session_prompt = "\n".join(chat_history + [f"User: {prompt}", "Assistant:"]) if memory else prompt
    payload = {"model": MODEL_NAME, "prompt": session_prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    answer = resp.json().get("response", "")
    if memory:
        chat_history.append(f"User: {prompt}")
        chat_history.append(f"Assistant: {answer}")
    return answer
