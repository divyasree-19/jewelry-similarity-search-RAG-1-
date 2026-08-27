# Cosine similarity
#        ↓
# Normalize vectors
#        ↓
# Inner Product search in FAISS

import faiss
import numpy as np
from embedding import embeddings

dimension=embeddings.shape[1]

print("embedding shape",embeddings.shape)
print("vector dimension",embeddings.shape[1])

#normalize product vectors
faiss.normalize_L2(embeddings)

#create faiss index using inner product
index = faiss.IndexFlatIP(dimension)

#add product vector to faiss
index.add(embeddings)
print("no. of vectors in faiss: ",index.ntotal)


'''
index = faiss.IndexFlatIP(dimension) ==> creates a FAISS index capable of storing:384-dimensional vectors


What does index.add() do?
This is the important line:

index.add(embeddings)

Our embeddings are:

Product 1 → Vector 1
Product 2 → Vector 2
Product 3 → Vector 3
...
Product 10 → Vector 10

FAISS now has:

FAISS Index

0 → Product 1 vector
1 → Product 2 vector
2 → Product 3 vector
3 → Product 4 vector
...
9 → Product 10 vector

That's why earlier we talked about positions.

FAISS knows the vectors by their index positions
'''