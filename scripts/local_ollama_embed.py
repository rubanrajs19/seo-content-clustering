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
        result = response.json()
        emb = result.get("embedding", [])
        if isinstance(emb, list) and len(emb) == 4096:
            print(f"✅ Ollama embedding OK: dim={len(emb)}")
            return emb
        else:
            print(f"❌ Ollama embedding has wrong dimension: {len(emb)}")
            return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None
