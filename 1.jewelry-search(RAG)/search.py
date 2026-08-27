import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from products import products
from embedding import embeddings

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# 1. Prepare product embeddings
# -----------------------------------

# Make a copy so we don't modify the original embeddings
product_vectors=np.asarray(embeddings,dtype="float32").copy()

#normalize for cosine similarity
faiss.normalize_L2(product_vectors)


# -----------------------------------
# 2. Create FAISS index
# -----------------------------------
dimension = product_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(product_vectors)

# -----------------------------------
# 3. User query
# -----------------------------------
query = "I want a gold ring with a diamond"
query_vector = model.encode(
    [query]
)
query_vector = np.asarray(
    query_vector,
    dtype="float32"
)

# Normalize query vector
faiss.normalize_L2(query_vector)


# -----------------------------------
# 4. Similarity Search
# -----------------------------------
k = 3
scores, positions = index.search(
    query_vector,
    k
)

# -----------------------------------
# 5. Display results
# -----------------------------------
print("\nSearch Query:")
print(query)

print("\nTop Results:\n")

for score, position in zip(scores[0], positions[0]):
     product = products[position]
     print(
        f"{product['name']} "
        f"(Similarity: {score:.4f})"
    )
