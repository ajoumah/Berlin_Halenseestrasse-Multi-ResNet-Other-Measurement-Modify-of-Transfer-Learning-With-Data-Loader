# 🗺️ Berlin_Halenseestrasse: Hierarchical Visual Place Recognition with ResNet-18 & Clustering

A full-stack implementation for **Hierarchical Visual Place Recognition (VPR)** using a combination of handcrafted features (HOG) and deep features (ResNet-18, ResNet-50, AlexNet-365). This project demonstrates how to identify, cluster, and classify visual scenes in an unsupervised or semi-supervised manner using transfer learning and PyTorch.

---

## 🧠 Hierarchical Method for Visual Place Recognition

This project employs a **hierarchical approach** for place recognition:

### Step 1: 🧮 Feature Extraction
- Use **Histogram of Oriented Gradients (HOG)** to capture low-level visual patterns.
- Use **ResNet-18 pretrained on Places365** to extract high-level semantic scene features.

### Step 2: 🔗 Similarity Computation
- Compute pairwise **cosine similarities** between feature vectors.
- Threshold-based grouping: if similarity ≥ 0.65, assign to same group (cluster).

### Step 3: 🧱 Hierarchical Clustering Logic
- Iterate over samples, assigning each to:
  - A new cluster if unclassified.
  - An existing cluster if it passes the similarity threshold.
- Track parent-child relationship in clustering (i.e., hierarchical formation).

### Step 4: 🏷️ Cluster Labeling & Dataset Generation
- Each cluster is assigned a unique binary string as its label.
- Images are saved in a structured folder hierarchy by label.

### Step 5: 🔁 Training with Deep CNN
- Augmented dataset is used to train/evaluate deep models like ResNet18/AlexNet.
- Hierarchical clustering serves as weak supervision for deep classifier.

---

## 📂 Features

- 📍 Hierarchical clustering for visual place discovery
- 🤖 Deep transfer learning with **ResNet-18 / ResNet-50 / AlexNet365**
- 🧮 HOG feature extraction for handcrafted baseline comparison
- 🗃️ Dynamic dataset generation with **auto-labeling & folder structuring**
- 🧪 Advanced data augmentation:
  - Gaussian blur
  - Salt & pepper noise
  - Contrast enhancement
  - Random masking
  - Rotation
- 📈 Visualization using `matplotlib`, `plotly`, and histograms
- 🔄 Training-ready PyTorch `DataLoader` for supervised/weakly-supervised learning

---

## 🧪 Datasets

### Primary
- **Berlin_Halenseestrasse** (custom or local dataset)

### Secondary (optional)
- **St. Lucia Urban Dataset**: Used for generalization and cross-dataset evaluation

---

## 📊 Visualization

- Cluster histogram with group sizes
- Feature similarity matrix (optional)
- Grid view of augmented image samples
- Batch previews from `DataLoader`

---

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/your-username/Berlin_Halenseestrasse-VPR.git
cd Berlin_Halenseestrasse-VPR
pip install -r requirements.txt
