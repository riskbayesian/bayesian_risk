#%%
import torch
import transformers
import os
import math
from transformers import pipeline
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("penguin.png")

feature_extractor = pipeline(
    model="dinov3-vitl16-pretrain-lvd1689m", # I store the model in a local directory
    task="image-feature-extraction",
)
features = feature_extractor(image)

features = torch.tensor(features)
features_no_batch = features.squeeze(0)

num_tokens = features_no_batch.shape[0]
num_patches_guess = int(math.sqrt(num_tokens))**2
h = w = int(math.sqrt(num_patches_guess))
num_special_tokens = num_tokens - num_patches_guess
patch_features = features_no_batch[num_special_tokens:, :]

pca = PCA(n_components=3)
features_pca = pca.fit_transform(patch_features.numpy())

pca_image = features_pca.reshape((h, w, 3))

pca_image_normalized = np.zeros_like(pca_image)
for i in range(3):
    channel = pca_image[:, :, i]
    pca_image_normalized[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())

pca_pil_image = Image.fromarray((pca_image_normalized * 255).astype(np.uint8))

pca_pil_image_nearest = pca_pil_image.resize(image.size, Image.Resampling.NEAREST)
pca_pil_image_bilinear = pca_pil_image.resize(image.size, Image.Resampling.BILINEAR)
pca_pil_image_bicubic = pca_pil_image.resize(image.size, Image.Resampling.BICUBIC)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# NEAREST 插值
axes[0, 1].imshow(pca_pil_image_nearest)
axes[0, 1].set_title("PCA Visualization - NEAREST")
axes[0, 1].axis('off')

# BILINEAR 插值
axes[1, 0].imshow(pca_pil_image_bilinear)
axes[1, 0].set_title("PCA Visualization - BILINEAR")
axes[1, 0].axis('off')

# BICUBIC 插值
axes[1, 1].imshow(pca_pil_image_bicubic)
axes[1, 1].set_title("PCA Visualization - BICUBIC")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
