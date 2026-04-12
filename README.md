# 🔍 Mini Google Lens using Caltech-256

A deep learning-based image classification system inspired by Google Lens.  
This project uses transfer learning with ResNet50 to classify images from the Caltech-256 dataset.

---

## 📌 Authors

- **Tirthankar Ghosh**
- **Richa Kumari**

---

## 🚀 Features

- 📦 Uses **Caltech-256 dataset (~30K images, 256 classes)**
- 🧠 Transfer learning with **ResNet50**
- 🎯 Achieves ~70% accuracy
- 🔄 Data augmentation for better generalization
- ⚡ Runs on **Apple Silicon (MPS GPU)** or CPU
- 🖼️ Image prediction (mini Google Lens-style)

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib
- PIL (Pillow)

---

## 📁 Project Structure
project/
│
├── dataset/
│ ├── 256_ObjectCategories/
│ └── 256_ObjectCategories.tar
│
├── model.pth
├── train.py
├── predict.py
└── README.md

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/mini-google-lens.git
cd mini-google-lens
