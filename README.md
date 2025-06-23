# 📍 Berlin_Halenseestrasse Visual Place Recognition Suite

A comprehensive framework for **Visual Place Recognition (VPR)**, scene classification, and clustering, applied to the **Berlin_Halenseestrasse** dataset. This project combines handcrafted features (HOG), deep features from ResNet/AlexNet, dynamic clustering, and automatic labeling, enabling scalable and accurate visual mapping for robotics, autonomous navigation, and geo-localization tasks.

---

## 🔧 Project Overview

This notebook integrates:

- 📸 Visual Feature Extraction using **HOG** and deep **ResNet18/ResNet50/AlexNet365** architectures
- 🧠 Transfer Learning using **Places365-pretrained CNNs**
- 🔁 Clustering based on **Cosine Similarity (≥ 0.65)** and distance thresholds
- 📊 Labeling and visualization using **matplotlib** and **plotly**
- 🗂️ Dataset creation with automatic **directory structuring** for each cluster
- 🔄 Data Augmentation: Blurring, Noise, Contrast Stretching, Random Masking, and Rotation
- ⚙️ End-to-End **PyTorch DataLoader** and classification setup

---

## 📂 Main Components

### 1. 🔍 Feature Extraction
- HOG descriptors for handcrafted local feature representation.
- Pretrained **ResNet18/ResNet50** and **AlexNet365** for high-level scene understanding.
- Custom logic for formatting and parsing image file names for sorted batch processing.

### 2. 🔗 Similarity and Clustering
- Cosine similarity-based clustering of HOG features.
- Dynamic thresholding to define similarity clusters.
- Classification labels are generated based on cluster membership.

### 3. 🧾 Dataset Labeling and Organization
- Each image is mapped to a new cluster label using a binary encoding format.
- New folders are created per label, and images are saved along with their augmented variants:
  - **Original**
  - **Blurred**
  - **Salt & Pepper Noise**
  - **Contrast Enhanced**
  - **Masked (random region cut)**
  - **Rotated (±45° random rotation)**

### 4. 🧠 Transfer Learning and Deep Classification
- Dataset is split using PyTorch `SubsetRandomSampler` into train/validation/test sets.
- ResNet18 Places365 model is loaded and frozen for inference.
- Optional fine-tuning capabilities can be integrated using custom `ImageClassificationBase` module.

---

## 🧪 Dataset Used

- **Berlin_Halenseestrasse**: Primary dataset of image sequences.
- **St. Lucia Urban Dataset** (optional): Used for cross-domain comparison and evaluation.

---

## 📈 Visualizations

- Cluster assignments plotted using:
  - Bar histograms with cluster sizes
  - Group-wise scatter plots of index vs. label
- Data loader batches visualized using `torchvision.make_grid()`

---

## 🧰 Requirements

Install required Python libraries:
```bash
pip install numpy pandas matplotlib scikit-image opencv-python scikit-learn torch torchvision plotly
