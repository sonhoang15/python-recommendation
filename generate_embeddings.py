import os
import json
import requests
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np

API_URL = os.getenv(
    "PRODUCT_API_URL",
    "http://localhost:8080/api/v1/product/read"
)
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

texts = []
valid_products = []

for p in products:
    text = build_text(p)
    if text:
        texts.append(text)
        valid_products.append(p)

products = valid_products

print(f"Embedding {len(products)} products...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
).astype(np.float32)

with open("products.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False, indent=2)

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": "all-MiniLM-L6-v2",
        "created_at": datetime.utcnow().isoformat(),
        "count": len(embeddings),
        "vectors": embeddings.tolist()
    }, f, indent=2)

print(f" DONE: {len(products)} products embedded")
