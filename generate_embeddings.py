import requests
import json
from sentence_transformers import SentenceTransformer

API_URL = "http://localhost:8080/api/v1/product/read"
model = SentenceTransformer("all-MiniLM-L6-v2")

res = requests.get(API_URL)
data = res.json()
products = data["DT"]  

texts = [
    f"{p.get('name', '')}. {p.get('description', '')}"
    for p in products
]

embeddings = model.encode(texts).tolist()

with open("products.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False, indent=2)

with open("embeddings.json", "w") as f:
    json.dump(embeddings, f)

print("DONE: Created products.json & embeddings.json")
