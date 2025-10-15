#%%
import faiss
import torch
# manipulated_points_f32 = np.ascontiguousarray(manipulated_points_np, dtype='float32')
# tree = faiss.IndexFlatL2(manipulated_points_f32.shape[1])
# tree.add(manipulated_points_f32)

# obj_points_f32 = np.ascontiguousarray(obj_points_np, dtype='float32')
# manipulated_points_f32 = np.ascontiguousarray(manipulated_points_np, dtype='float32')

# faiss.omp_set_num_threads(6)  

# D, _ = tree.search(obj_points_np, 1)
# min_distances_environment = np.sqrt(D).squeeze()

import numpy as np
import time
from scipy.spatial import KDTree

d = 3                           # dimension
nb = 100                      # database size
nq = 100                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

tnow = time.time()
tree = KDTree(xb)
query = tree.query(xq, k=1, workers=-1)
print(f"Time taken (scipy): {time.time() - tnow}")

import faiss                     # make faiss available

res = faiss.StandardGpuResources()  # use a single GPU

## Using a flat index

# tnow = time.time()
# torch.cuda.synchronize()

# index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

# # make it a flat GPU index
# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

# gpu_index_flat.add(xb)         # add vectors to the index
# print(gpu_index_flat.ntotal)

# k = 4                          # we want to see 4 nearest neighbors
# D, I = gpu_index_flat.search(xq, k)  # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries

# torch.cuda.synchronize()
# print(f"Time taken: {time.time() - tnow}")

## Using an IVF index
nlist = 100
tnow = time.time()
torch.cuda.synchronize()
quantizer = faiss.IndexFlatL2(d)  # the other index
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search

# make it an IVF GPU index
gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)

assert not gpu_index_ivf.is_trained
gpu_index_ivf.train(xb)        # add vectors to the index
assert gpu_index_ivf.is_trained

gpu_index_ivf.add(xb)          # add vectors to the index
print(gpu_index_ivf.ntotal)

k = 1                          # we want to see 4 nearest neighbors
D, I = gpu_index_ivf.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

torch.cuda.synchronize()
print(f"Time taken: {time.time() - tnow}")


# tnow = time.time()
# torch.cuda.synchronize()
# index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

# index_flat.add(xb)         # add vectors to the index
# print(index_flat.ntotal)

# k = 4                          # we want to see 4 nearest neighbors
# D, I = index_flat.search(xq, k)  # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries

# torch.cuda.synchronize()
# print(f"Time taken (cpu): {time.time() - tnow}")