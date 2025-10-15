# 🌍 Open World Risk - Risk Assessment Pipeline

> A complete machine learning pipeline for assessing risk in open-world environments using RGB-D vision and Bayesian inference.

This system processes RGB-D video data to generate spatial risk assessments by combining **prior knowledge** (common-sense understanding) with **likelihood observations** (real-world data) through Bayesian posterior computation.

---

## 📊 Pipeline Overview

The system follows a four-stage Bayesian inference pipeline:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   📹 Video   │ →  │  🧠 Prior    │ →  │  📊 Like-    │ →  │  🔗 Post-    │
│  Processing  │    │   Training   │    │   lihood     │    │   erior      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       ↓                    ↓                    ↓                    ↓
video_processing/   common_sense_priors/  data_deriv_likelihood/  posterior/
```

**Bayesian Framework:**  
`P(risk | observations) ∝ P(observations | risk) × P(risk)`

## 🏗️ Pipeline Components

### 📹 **1. Video Processing** (`video_processing/`)

Processes RGB-D video data to extract object-level features and risk information.

**Key Scripts:**
- `process_video.py` - Main processing script with object detection
- `process_video_utils.py` - Utility functions
- `images_to_video.py` - Frame-to-video conversion

> 📥 **Input:** RGB-D video data (.npz files with RGB, depth, pose)  
> 📤 **Output:** Object masks, features, manipulation scores per frame

---

### 🧠 **2. Prior Training** (`common_sense_priors/`)

Trains models to encode common-sense knowledge about object risk.

**Key Scripts:**
- `train_prior.py` - Prior model training
- `model_prior.py` - Model architecture
- `dataset_prior.py` - Dataset handling
- `batch_prior_over_folders.py` - Batch processing

> 📥 **Input:** Synthetic risk data, object labels  
> 📤 **Output:** Trained prior models with risk knowledge

---

### 📊 **3. Likelihood Training** (`data_deriv_likelihood/`)

Extracts risk likelihood from real video observations.

**Key Scripts:**
- `inference_riskfield_video.py` - Main inference script
- `inference_riskfield_video_lite.py` - Lightweight version
- `build_lookup_likelihood.py` - Lookup table builder
- `likelihood_from_single_frame.py` - Single frame processor

> 📥 **Input:** Processed video from `video_processing/`  
> 📤 **Output:** Risk likelihood maps per frame

---

### 🔗 **4. Posterior Combination** (`posterior/`)

Combines prior knowledge with likelihood observations for final risk assessment.

**Key Scripts:**
- `combine_likelihood_and_prior.py` - Basic combination (single object)
- `combine_likelihood_and_prior_tree.py` - Advanced with depth modulation
- `combine_likelihood_and_prior_tree_.py` - Multi-object without depth
- `posterior_to_ply.py` - 3D point cloud generator

> 📥 **Input:** Prior risk maps + likelihood risk maps  
> 📤 **Output:** Final posterior risk assessments (2D + 3D)

## 🚀 Getting Started

### 📦 Step 1: Pull Docker Images

```bash
docker pull bayesianrisk/open_world_risk_main:latest &
docker pull bayesianrisk/open_world_risk_hoist_former:latest &
```

### 🌐 Step 2: Create Docker Network

```bash
docker network create open_world_risk_net
```

### 🖥️ Step 3: Launch Containers

**Start Hoist Former Container (Server)**
```bash 
docker run --rm \
     --runtime=nvidia --gpus all \
     --network open_world_risk_net \
     -p 5002:5000 \
     -v $(pwd):/workspace -w /workspace \
     -e PYTHONWARNINGS=ignore::FutureWarning \
     -e PYTHONPATH=/workspace/open_world_risk/hoist_former/Mask2Former \
     --name hoist_former \
     bayesianrisk/open_world_risk_hoist_former:latest \
     /bin/bash -c "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
                   conda activate hoist2 && \
                   cd /workspace/open_world_risk/hoist_former/Mask2Former && \
                   pip install -qq -r requirements.txt && \
                   cd mask2former/modeling/pixel_decoder/ops && \
                   sh make.sh > /dev/null 2>&1 && \
                   pip install -qq setuptools==59.5.0 && \
                   cd /workspace && \
                   python -c 'import mask2former; print(\"🚀 Mask2Former ready!\")' && \
                   python open_world_risk/hoist_former/api/endpoints.py"
```

**Start Main Container (Interactive)**
```bash 
docker run -it --rm \
     --runtime=nvidia --gpus all \
     --shm-size=8gb \
     --network open_world_risk_net \
     -p 5001:5000 \
     -v $(pwd):/workspace -w /workspace \
     -e PYTHONWARNINGS=ignore::UserWarning,ignore::FutureWarning \
     --name main \
     bayesianrisk/open_world_risk_main:latest \
     bash
```

**Activate Python Environment (inside main container)**
```bash
conda activate feature_splatting2
```

---

## 🔄 Running the Complete Pipeline

Execute the following steps in order (all commands run inside the main container):
0. **Download Docker Images and Run Containers
***Pull images from Docker Hub***
```bash
docker pull bayesianrisk/open_world_risk_main:latest &
docker pull bayesianrisk/open_world_risk_hoist_former:latest &
```
***Create Network for Containers to Communicate***
```bash
docker network create open_world_risk_net
```
***Run hoist_former container as server***
```bash 
docker run --rm \
     --runtime=nvidia --gpus all \
     --network open_world_risk_net \
     -p 5002:5000 \
     -v $(pwd):/workspace -w /workspace \
     -e PYTHONWARNINGS=ignore::FutureWarning \
     -e PYTHONPATH=/workspace/open_world_risk/hoist_former/Mask2Former \
     --name hoist_former \
     bayesianrisk/open_world_risk_hoist_former:latest \
     /bin/bash -c "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate hoist2 && cd /workspace/open_world_risk/hoist_former/Mask2Former && pip install -qq -r requirements.txt && cd mask2former/modeling/pixel_decoder/ops && sh make.sh > /dev/null 2>&1 && pip install -qq setuptools==59.5.0 && cd /workspace && python -c 'import mask2former; print(\"🚀🚀🚀 Mask2Former import successful! 🚀🚀🚀\")' && python open_world_risk/hoist_former/api/endpoints.py"
```
***Run main container interactively (all subsequent commands will be executed in the main container)***
```bash 
docker run -it --rm \
     --runtime=nvidia --gpus all \
     --shm-size=8gb \
     --network open_world_risk_net \
     -p 5001:5000 \
     -v $(pwd):/workspace -w /workspace \
     -e PYTHONWARNINGS=ignore::UserWarning,ignore::FutureWarning \
     --name main \
     bayesianrisk/open_world_risk_main:latest \
     bash
```
***In main container activate the python environment***
```bash
conda activate feature_splatting2
```

### 📹 **Step 1: Process Video Data**
Extract object features and risk information from RGB-D video:
```bash
python open_world_risk/video_processing/process_video.py
```

### 🧠 **Step 2: Train Prior Models**
Train models on common-sense risk knowledge:
```bash
python open_world_risk/common_sense_priors/train_prior.py
```

### 📊 **Step 3: Generate Likelihood from Real Data**
Extract risk likelihood from processed video:
```bash
python open_world_risk/data_deriv_likelihood/inference_riskfield_video.py
```

### 🔗 **Step 4: Combine Prior + Likelihood**
Compute final posterior risk assessments:
```bash
python open_world_risk/posterior/combine_likelihood_and_prior_tree.py
```

### 🎨 **Step 5: Generate 3D Risk Visualization**
Convert posterior to 3D point clouds:
```bash
python open_world_risk/posterior/posterior_to_ply.py
```

---

## ⚙️ Configuration

Each pipeline component uses YAML configuration files for easy customization:

| Component | Config File |
|-----------|------------|
| 📹 Video Processing | `video_processing/process_video_config.yaml` |
| 🧠 Prior Training | `common_sense_priors/train_prior_config.yaml` |
| 📊 Likelihood | `data_deriv_likelihood/infer_risk_field_vid_cfg.yaml` |
| 🔗 Posterior | `posterior/combine_tree_config.yaml` |

### 🔑 **Important: WandB Configuration**

Before running the training scripts, you **must** configure your Weights & Biases credentials:

1. **Create a WandB account** at [wandb.ai](https://wandb.ai) (if you don't have one)
2. **Update the following config files** with your WandB username:
   - `open_world_risk/train/train_config.yaml` (line 17)
   - `open_world_risk/common_sense_priors/train_prior_config.yaml` (line 17)

Replace `"userA"` with your actual WandB username:
```yaml
wandb:
  project: "open_world_risk_0"
  run_name: "rf_bezier_supervision"
  entity: "your_wandb_username"  # ← Replace "userA" with YOUR username
```

> 💡 **Tip:** Set `entity: null` if you're not using a WandB team/organization.

---

## ✨ Key Features

- 🎯 **Multi-object Risk Assessment** - Handles multiple objects simultaneously
- 📏 **Depth-aware Processing** - Incorporates depth information for 3D risk modeling
- 🎨 **3D Visualization** - Generates PLY point clouds for risk visualization
- 🧩 **Modular Design** - Each component can be used independently

---

## 📋 Requirements

- 🐋 Docker (to download containers with environment)
- 🖥️ NVIDIA GPU (CUDA-compatible) required for training/inference  
  _(We used an RTX 4090 for our experiments.)_


