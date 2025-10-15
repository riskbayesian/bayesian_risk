from setuptools import setup, find_packages

setup(
    name="open_world_risk",
    version="0.1.0",
    description="Open World Risk Assessment using Computer Vision and ML",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "numpy>=1.21.0",
        "opencv-python>=4.8.0",
        "open3d>=0.17.0",
        "Pillow>=9.0.0",
        "scipy>=1.9.0",
        "h5py>=3.8.0",
        "tqdm>=4.64.0",
        "einops>=0.6.0",
        "transformers>=4.30.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
        "full": [
            "taichi>=1.4.0",
            "wandb>=0.15.0",
            "vuer>=0.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)