# Prior Model Training Guide

This guide provides step-by-step instructions for training and evaluating the Prior Model in the Open World Risk system.

## 🚀 Quick Start

Follow these three steps to train and evaluate your Prior Model:

---

## 📊 Step 1: Build Dataset

```bash
python open_world_risk/common_sense_priors/build_dataset.py
```

**What this does:**
- Generates the dataset for training the Prior Model
- Uses `build_dataset_config.yaml` from the local directory and copies it to output directory
- Configuration file contains training parameters and is automatically saved

---

## 🎯 Step 2: Train Prior Model

```bash
python open_world_risk/common_sense_priors/train_prior.py
```

**Configuration:**
- Uses `train_prior_config.yaml` in the local directory
- Trains the model defined in `model_prior.py` (same directory)
- Follows the configuration specified in the YAML file

---

## 🔍 Step 3: Evaluate Model

```bash
python open_world_risk/train/inference_riskfield_video.py
```

**Configuration:**
- Uses `infer_risk_field_vid_cfg.yaml` in the `open_world_risk/train` directory
- **Supported Input Formats:** `.mov` and `.png` files
- Specify input via `image_a` parameter

---

## ⚠️ Important Prerequisites

> **Note:** Ensure you have a trained likelihood model as defined in `open_world_risk/utils/model.py` before running evaluation, as the inference pipeline requires both prior and likelihood models.

---

## 📁 File Structure

```
open_world_risk/common_sense_priors/
├── build_dataset.py              # Dataset generation script
├── train_prior.py                # Training script
├── model_prior.py                # Prior model definition
├── build_dataset_config.yaml     # Dataset configuration (auto-generated)
└── train_prior_config.yaml       # Training configuration

open_world_risk/train/
├── inference_riskfield_video.py  # Inference script
└── infer_risk_field_vid_cfg.yaml # Inference configuration

```
