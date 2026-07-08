# 🔍 Mini Google Lens

A multimodal AI-powered visual search engine inspired by **Google Lens**. This project combines deep learning, vision-language models, OCR, semantic image retrieval, object localization, and LLM reasoning to understand images beyond simple classification.

---

# 📌 Authors

- **Tirthankar Ghosh**
- **Richa Kumari**

---

# 🚀 Features

- 🖼️ Upload any image for visual analysis
- 🧠 OpenCLIP-based scene understanding
- 📝 BLIP image caption generation
- 🔤 OCR text extraction using EasyOCR
- 🎯 ResNet50 classifier trained on Caltech-256
- 🔍 Similar image retrieval using CLIP embeddings
- 🌐 Internet image search (DuckDuckGo)
- 🛍️ Product search links (Amazon & Flipkart)
- 📍 Open-vocabulary object localization using OWLv2
- 🤖 AI-powered reasoning using Qwen 2.5
- 💬 Natural language image search
- 🎨 Google Lens-inspired responsive UI

---

# 🧠 AI Pipeline

```
                Upload Image
                      │
                      ▼
            OpenCLIP Scene Detection
                      │
                      ▼
             BLIP Image Captioning
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     EasyOCR                ResNet50
   (Documents)         (Caltech-256 Classes)
          │                       │
          └───────────┬───────────┘
                      ▼
              Qwen 2.5 Reasoning
                      │
     ┌────────┬─────────────┬───────────┐
     ▼        ▼             ▼           ▼
 Shopping   Web Images   CLIP Search   OWLv2
  Links      Search      Similarity  Localization
```

---

# 📊 Models Used

| Model | Purpose |
|--------|---------|
| ResNet50 | Image Classification |
| OpenCLIP | Scene Understanding |
| BLIP | Image Captioning |
| EasyOCR | Text Extraction |
| Qwen 2.5-1.5B | Multimodal Reasoning |
| OWLv2 | Object Localization |
| CLIP | Similar Image Retrieval |

---

# 🛠 Tech Stack

### Backend

- FastAPI
- Python
- PyTorch
- Hugging Face Transformers
- Torchvision

### AI Models

- ResNet50
- OpenCLIP
- BLIP
- EasyOCR
- Qwen 2.5
- OWLv2

### Frontend

- HTML5
- CSS3
- JavaScript
- Jinja2 Templates

### Libraries

- Pillow
- NumPy
- Scikit-learn
- FAISS / CLIP Embeddings
- DuckDuckGo Search

---

# 📂 Project Structure

```text
Mini-Google-Lens/
│
├── backend/
│   ├── main.py
│   ├── classifier.py
│   ├── captioning.py
│   ├── clip_router.py
│   ├── localization.py
│   ├── retrieval.py
│   ├── similarity.py
│   ├── ocr.py
│   ├── chat.py
│   ├── shopping.py
│   ├── web_search.py
│   └── category.py
│
├── static/
│
├── templates/
│
├── dataset/
│
├── model.pth
├── embeddings.npy
├── image_index.faiss
├── image_paths.pkl
├── classes.json
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/mini-google-lens.git
cd Mini-Google-Lens
```

### Create a virtual environment

```bash
conda create -n mini_lens python=3.11
conda activate mini_lens
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
uvicorn backend.main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

---

# 🎯 Sample Capabilities

- Detect everyday objects
- Recognize products
- Generate image descriptions
- Extract text from documents
- Localize detected objects
- Retrieve visually similar images
- Search visually similar images on the web
- Provide shopping links for products

---

# 📈 Results

- ResNet50 trained on **Caltech-256**
- ~70% classification accuracy
- Zero-shot scene understanding using OpenCLIP
- Accurate object localization with OWLv2
- AI-assisted semantic reasoning with Qwen
- Supports both document understanding and object recognition

---

# 🔮 Future Improvements

- Multi-object localization
- Visual Question Answering (VQA)
- Voice-based image search
- Real-time webcam support
- Mobile deployment
- Fine-tuned multimodal reasoning model

---

# 📜 License

This project is developed for academic and educational purposes.