# 🧠 EEG-ChebNet: Spectral Graph Learning for Alcoholism Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![GNN Library](https://img.shields.io/badge/PyG-PyTorch%20Geometric-green.svg)](https://www.pyg.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20EEG-blue.svg)](https://www.kaggle.com/datasets/nnair25/Alcoholics)
![Accuracy](https://img.shields.io/badge/Accuracy-75%25-brightgreen)

> **A Novel Spatio-Temporal Graph Convolutional Network using Chebyshev Polynomials for EEG Classification.**

---

## 📖 Overview

This repository implements a **Spectral Spatio-Temporal Graph Neural Network (ST-GNN)** designed to detect alcoholism from raw multi-channel EEG signals.

Unlike traditional methods that rely on manual feature extraction or simple 1D-CNNs, this framework models the brain as a complex network. It combines **1D Convolutional Neural Networks** (for temporal feature extraction) with **Chebyshev Spectral Graph Convolutions (ChebNet)** (for spatial connectivity), allowing the model to capture long-range dependencies between brain regions using higher-order polynomials ($K=3$).

## 🚀 Key Novelty & Features

This architecture addresses the limitations of standard GCNs in modeling brain connectivity:

1.  **Spectral Graph Convolutions (ChebNet):** Unlike standard GCNs that only look at immediate neighbors, this model uses **3rd-order Chebyshev polynomials** ($K=3$). This allows the model to "see" neighbors-of-neighbors in a single layer, effectively modeling signal propagation across the scalp.
2.  **Hybrid ST-Architecture:**
    * **Temporal:** 1D-CNNs extract high-frequency features (e.g., Gamma/Beta bands) from raw time series.
    * **Spatial:** ChebNet layers fuse these features based on the functional connectivity of the electrodes.
3.  **Robust Regularization:** Integrated Dropout and Batch Normalization layers prevent overfitting on the noisy EEG dataset.

---

## 📂 Dataset

The model is trained on the **UCI EEG Alcoholism Database**.

- **Dataset Link:** [Kaggle: EEG Alcoholics Dataset](https://www.kaggle.com/datasets/nnair25/Alcoholics)
- **Description:** The dataset contains EEG recordings from 122 subjects (Alcoholic and Control groups).
- **Format:** 64-channel EEG time-series sampled at 256 Hz (1-second trials).

### ⚙️ Directory Structure
To run this code, please download the dataset from the link above and organize your directory as follows:

```text
/repo-root
│
├── eeg.ipynb           # Main training and evaluation notebook
├── SMNI_CMI_TRAIN/     # Directory containing training .csv files
│   ├── Data1.csv
│   └── ...
└── SMNI_CMI_TEST/      # Directory containing testing .csv files
    ├── Data3.csv
    └── ...
