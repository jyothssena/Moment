import json
import numpy as np # type: ignore
import faiss # type: ignore

INPUT_FILE    = "data/processed/moments_embedded.json"
INDEX_FILE    = "data/processed/moments.index"
MAP_FILE      = "data/processed/index_map.json"

with open(INPUT_FILE, "r") as f:
    moments = json.load(f)

print(f"Loaded {len(moments)} moments.")

# Extract embeddings into a numpy array
vectors = np.array([m["embedding"] for m in moments], dtype="float32")

# Normalize for cosine similarity
faiss.normalize_L2(vectors)

# Build the index
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors)

print(f"Index built. Total vectors: {index.ntotal}, Dimensions: {dimension}")

# Save the index to disk
faiss.write_index(index, INDEX_FILE)
print(f"Saved index to {INDEX_FILE}")

# Build and save the position-to-metadata map
index_map = [
    {
        "position": i,
        "moment_id": m["interpretation_id"],
        "user_id": m["user_id"],
        "book_id": m["book_id"]
    }
    for i, m in enumerate(moments)
]

with open(MAP_FILE, "w") as f:
    json.dump(index_map, f, indent=2)

print(f"Saved index map to {MAP_FILE}")

# Quick test query — take the first vector, find top 5 similar
print("\nRunning test query (top 5 results for moments[0])...")
query = vectors[0:1]
distances, indices = index.search(query, 5)

for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
    entry = index_map[idx]
    print(f"  Rank {rank+1}: {entry['moment_id']} | user: {entry['user_id']} | score: {score:.4f}")