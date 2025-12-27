from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Python Recommendation Service")

with open("products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

with open("embeddings.json", "r") as f:
    embeddings = np.array(json.load(f))

model = SentenceTransformer("all-MiniLM-L6-v2")

class RecommendRequest(BaseModel):
    text: str
    top_k: int = 5

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        query_emb = model.encode([req.text])
        sim = cosine_similarity(query_emb, embeddings)[0]

        top_idx = np.argsort(sim)[::-1][:req.top_k]

        results = []
        for idx in top_idx:
            product = products[idx]
            results.append({
                "id": product.get("id", product.get("_id")),
                "name": product.get("name"),
                "score": float(sim[idx]),
                "description": product.get("description"),
                "price": product.get("price_min"),
                "image": product.get("thumbnail"),
            })

        return {
            "EC": 0,
            "EM": "OK",
            "DT": results
        }

    except Exception as e:
        return {
            "EC": 1,
            "EM": str(e),
            "DT": []
        }
