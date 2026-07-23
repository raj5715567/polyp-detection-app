# 🩺 AI-Based Polyp Detection and Risk Assessment

> An AI-powered medical image analysis system for **automatic colon polyp detection, semantic segmentation, and risk assessment** using **DeepLabV3 with ResNet50**. The project includes a **Streamlit web application** for real-time prediction and visualization.

---

## 📌 Overview

Colorectal cancer is one of the leading causes of cancer-related deaths worldwide. Most colorectal cancers develop from **colon polyps**, making their early detection extremely important.

This project leverages **Deep Learning** and **Computer Vision** techniques to automatically detect and segment colon polyps from colonoscopy images. After segmentation, the system estimates the **relative affected area** and performs **risk assessment**, providing an intelligent diagnostic support tool for healthcare professionals.

---

## 🚀 Features

- ✅ Automatic Colon Polyp Detection
- ✅ Semantic Segmentation using DeepLabV3
- ✅ ResNet50 Backbone for Feature Extraction
- ✅ Pixel-wise Segmentation Mask Generation
- ✅ Relative Polyp Area Calculation
- ✅ Automated Risk Assessment
- ✅ Streamlit Web Interface
- ✅ Real-time Image Upload & Prediction
- ✅ Highlighted Polyp Visualization
- ✅ Downloadable Medical Report

---

# 📂 Project Structure

# 📂 Project Structure

```text
AI-Based-Polyp-Detection/
│
├── app3.py            # Main Streamlit application
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

---

# 🏗️ System Architecture

```
Input Colonoscopy Image
            │
            ▼
     Image Preprocessing
            │
            ▼
 DeepLabV3 + ResNet50 Backbone
            │
            ▼
  Semantic Segmentation Mask
            │
            ▼
 Relative Area Calculation
            │
            ▼
    Risk Assessment
            │
            ▼
 Highlighted Visualization
            │
            ▼
 Downloadable Medical Report
```

---

# 🤖 Model Storage

To keep the GitHub repository lightweight, the trained **DeepLabV3 model** is **not stored in this repository**.

When the application starts, it automatically downloads the pretrained model from **Google Drive** and loads it into memory for prediction. This approach avoids GitHub's file size limitations while ensuring the latest trained model is used during inference.

# 🧠 Model Architecture

The project uses **DeepLabV3** for semantic segmentation.

### Backbone
- ResNet50 (ImageNet Pretrained)

### Segmentation Module
- Atrous Spatial Pyramid Pooling (ASPP)

### Activation
- Sigmoid

### Optimizer
- Adam

### Loss Function
- Binary Crossentropy

---

# 📊 Dataset

**Dataset Used:** Kvasir-SEG

Dataset contains:

- 528 Colonoscopy Images
- 528 Ground Truth Masks

### Preprocessing

- Resize to **256 × 256**
- Pixel Normalization
- Image-Mask Alignment
- Train / Validation / Test Split

---

# ⚙️ Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Web Framework | Streamlit |
| Model | DeepLabV3 |
| Backbone | ResNet50 |

---

# 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Dice Coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score

### Model Performance

| Metric | Value |
|---------|--------|
| Accuracy | **98.11%** |
| Dice Coefficient | **0.8580** |
| IoU | **0.8724** |
| Precision | **0.9330** |
| Recall | **0.9298** |
| F1 Score | **0.9279** |

---

# ⚠️ Risk Assessment

After segmentation, the system calculates the **relative polyp area**.

The detected region is classified into:

- 🟢 Low Risk
- 🟡 Moderate Risk
- 🔴 High Risk

This assists clinicians in preliminary medical evaluation.

---

# 💻 Streamlit Web Application

The web application allows users to:

- Upload colonoscopy images
- Generate segmentation masks
- View highlighted predictions
- Calculate polyp area
- Perform automated risk assessment
- Download prediction reports

---

# 📷 Sample Results

### Original Image

> Upload a colonoscopy image for analysis.

### Segmentation Mask

> DeepLabV3 generates a pixel-wise binary segmentation mask.

### Highlighted Prediction

> Detected polyp region is highlighted on the original image.

---

# 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/raj5715567/polyp-detection-app.git
```

Navigate to the project directory

```bash
cd polyp-detection-app
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 🌐 Live Demo

🔗 https://polyp-detection-app-7.streamlit.app/

---

# 📌 Future Improvements

- Real-time Colonoscopy Video Analysis
- Transformer-based Segmentation Models
- Explainable AI (XAI)
- Hospital Information System Integration
- Clinical Validation with Larger Datasets
- Mobile Application Development

---

# 📚 References

1. Litjens et al., *Medical Image Analysis*, 2017.
2. Chen et al., *ECCV*, 2018.
3. Ronneberger et al., *MICCAI*, 2015.
4. Jha et al., *Multimedia Modeling*, 2020.
5. He et al., *CVPR*, 2016.

---

# 👨‍💻 Author

**Raj Kumar**

B.Tech Computer Science Engineering (AI & ML)

Jawaharlal Nehru Government Engineering College, Sundernagar

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is developed for educational and research purposes.
