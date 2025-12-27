import os
import json
import requests
from sentence_transformers import SentenceTransformer
from datetime import datetime

API_URL = os.getenv("PRODUCT_API_URL", "http://localhost:8080/api/v1/product/read")
TIMEOUT = 10

print("Fetching products from:", API_URL)

res = requests.get(API_URL, timeout=TIMEOUT)
res.raise_for_status()

data = res.json()
products = data.get("DT", [])

def build_text(p):
    name = p.get("name") or ""
    desc = p.get("description") or ""
    return f"{name}. {desc}".strip()

texts = [build_text(p) for p in products]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True).tolist()

with open("products.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False, indent=2)

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": "all-MiniLM-L6-v2",
        "created_at": datetime.utcnow().isoformat(),
        "count": len(embeddings),
        "vectors": embeddings
    }, f, indent=2)

print(f"DONE: {len(products)} products embedded")
