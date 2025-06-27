import requests

def get_mistral_embedding(text):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "avr/sfr-embedding-mistral",
        "prompt": text
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['embedding']
    except Exception as e:
        print(f"Ollama error: {e}")
        return None
