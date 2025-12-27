from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Python Recommendation Service")

with open("products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

with open("embeddings.json", "r") as f:
    data = json.load(f)
    embeddings = np.array(data["vectors"], dtype=np.float32)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

class RecommendRequest(BaseModel):
    text: str
    top_k: int = 5

def cosine_similarity_np(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        query_emb = model.encode([req.text]).astype(np.float32)
        sim = cosine_similarity_np(query_emb, embeddings)[0]

        top_k = min(req.top_k, 20)
        top_idx = np.argsort(sim)[::-1][:top_k]

        results = []
        for idx in top_idx:
            p = products[idx]
            results.append({
                "id": p.get("id") or p.get("_id"),
                "name": p.get("name"),
                "score": float(sim[idx]),
                "description": p.get("description"),
                "price": p.get("price_min"),
                "image": p.get("thumbnail"),
            })

        return { "EC": 0, "EM": "OK", "DT": results }

    except Exception as e:
        print(" ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
