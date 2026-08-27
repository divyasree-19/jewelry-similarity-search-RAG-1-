# from sentence_transformers import SentenceTransformer
# from products import products
# import numpy as np

# #Load the embedding model
# model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# #store embeddings
# product_embedding=[]

# for product in products:
#     vector=model.encode(product["description"])
#     product_embedding.append(vector)

#     print(f"product:{product['name']}")
#     print(f"vector dimension: {len(vector)}")
#     print(f"first 5 calues: {vector[:10]}")
#     print("-"*50)



from sentence_transformers import SentenceTransformer
import numpy as np

from products import products

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Extract product descriptions
texts = [
    product["description"]
    for product in products
]

# Generate embeddings
embeddings = model.encode(texts)

# Convert to NumPy array
embeddings = np.asarray(embeddings, dtype="float32")


print("Embedding shape:", embeddings.shape)
print("Number of products:", len(products))
print("Dimensions per vector:", embeddings.shape[1])

print("\nFirst product:")
print(products[0]["name"])

print("\nFirst 5 values of its embedding:")
print(embeddings[0][:5])

'''
.shape --> .shape is an attribute of a NumPy array (or a PyTorch/TensorFlow tensor) that returns a tuple showing the size of the array along each of its dimensions.
AI Mode conversation: what is numpy array in pythonYou said: what is numpy array in pythonwhat is numpy array in pythonA NumPy array is a highly efficient, multi-dimensional data structure provided by the NumPy Library in Python for handling numerical data. Known officially as an ndarray (n-dimensional array), it serves as the foundational building block for data science, machine learning, and scientific computing.Key FeaturesHomogeneous: All elements must share the exact same data type, such as all integers or all floats.Contiguous Memory: Blocks are stored sequentially in memory, enabling ultra-fast processing speeds.Vectorized Operations: Mathematical operations apply directly to entire arrays without needing slow loops.Multi-Dimensional: Structures scale seamlessly from 1D vectors to 2D matrices and higher-dimensional tensors.NumPy Arrays vs. Python ListsWhile they look similar, their performance and behavior under the hood are completely different:FeaturePython ListsNumPy ArraysData TypesHeterogeneous (can mix types)Homogeneous (strictly single type)SpeedSlower due to non-contiguous storageUp to 50x faster, optimized in CMemoryHigh consumption (stores pointers)Highly efficient, compact layoutMultiplicationDuplicates the items [1, 2] * 2 -> [1, 2, 1, 2]Multiplies element-wise [1, 2] * 2 -> [2, 4]Basic ImplementationTo use a NumPy array, you must first import the library. Review the foundational creation syntax below, documented comprehensively on W3Schools NumPy Creation Guide:pythonimport numpy as np

# 1-Dimensional Array (Vector)
vector = np.array([1, 2, 3, 4])

# 2-Dimensional Array (Matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Element-wise operation example
print(vector * 2)  # Output: [2, 4, 6, 8]
Use code with caution.If you want to dive deeper, let me know if you would like to see:How to index and slice data out of these arrays.Built-in helpers like np.zeros or np.arange for generating data.How to reshape dimensions for machine learning workflows.YouTube·Bro CodeStart using NumPy in 5 minutes! 🔢yo what's going on everybody in this video I'm going to teach you everything you need to know to start working with NumPy. so sit ...5:08YouTube·TeluskoWhat is Numpy and Why?welcome back aliens my name is Davin Reddi and in this video we'll talk about numpy. or numpy. now basically numpy stands for nume...9:25YouTube·Code with Josh12 Minutes to Learn NumPy (Python): Arrays, Reshape, Arange ...my goal in this video is to try and teach you numpy in like 12 minutes now what is numpy numpy really stands for numerical Python ...17:15Show allembeddings.shape[1] what is shape show with an exampleIn Python, .shape is an attribute of a NumPy array (or a PyTorch/TensorFlow tensor) that returns a tuple showing the size of the array along each of its dimensions.When you see embeddings.shape[1], it is extracting the size of the second dimension (index 1) of that array. In machine learning, this specific dimension almost always represents the embedding dimension (the length of the vector representing a word or sentence).Understanding .shape by IndexIf an array has two dimensions, its .shape will return a tuple with two numbers: (rows, columns).embeddings.shape: Returns the full shape tuple, like (3, 768).embeddings.shape[0]: Represents the number of rows (usually the number of sentences, words, or batch size).embeddings.shape[1]: Represents the number of columns (the hidden size or embedding vector length).

Why float32?
is important because FAISS commonly works with 32-bit floating-point vectors.
So we're preparing the data in the format FAISS expects.


output:
Embedding shape: (10, 384)
Number of products: 10
Dimensions per vector: 384

First product:
Gold Diamond Ring

First 5 values of its embedding:
[... ... ... ... ...]



What is the dimensionality of each vector?
Our vectors are: 384-dimensional
Therefore:
dimension = 384

Then we'll create a FAISS index.
For example, if we want Euclidean/L2 search:

  1.index = faiss.IndexFlatL2(384)

Or if we want inner-product search:

  2.index = faiss.IndexFlatIP(384)

Notice:
IndexFlatL2(384)
             ↑
         384 dimensions
'''