# 🎤 Speech Emotion Recognition using CNN-BiLSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green)](https://librosa.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🎓 Final Year Project – BTS Artificial Intelligence Development  
> 👨‍💻 Author: **Ilyass AIT CHEIKH**  
> 📅 Academic Year: 2024–2025  

---

## 🚀 Project Overview

This project presents a **Deep Learning system for Speech Emotion Recognition (SER)** using a hybrid **CNN-BiLSTM architecture**.

The system classifies speech signals into **7 emotions**:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

### 🎯 Final Performance
- ✅ Accuracy: **91%**
- ✅ Macro F1-score: **0.91**
- ✅ Strong multi-dataset generalization

A **Flask web application** is included for real-time emotion prediction.

---

## 📊 Datasets

The model was trained on four public datasets:

- CREMA-D  
- RAVDESS  
- TESS  
- SAVEE  

⚠ Raw audio files are NOT included in this repository due to size limitations.

---

## 🔧 Processing Pipeline

### 1️⃣ Audio Preprocessing
- Fixed duration: 2.5 seconds  
- Offset trimming  
- Amplitude normalization  

### 2️⃣ Data Augmentation
- Gaussian noise  
- Time stretching  
- Pitch shifting  
- Time shifting  
- Volume scaling  
- Speed perturbation  

### 3️⃣ Feature Extraction
- 40 MFCC  
- Mel Spectrogram  
- ZCR  
- RMS Energy  
- Chroma STFT  
- Spectral Contrast  
- Tonnetz  

### 4️⃣ Class Balancing
- SMOTE  

### 5️⃣ Feature Scaling
- StandardScaler  

---

## 🧠 Model Architecture

```
Input (195,1)
→ Conv1D (128) + BN + Pool + Dropout
→ Conv1D (256) + Pool + Dropout
→ Conv1D (512) + Pool + Dropout
→ Bidirectional LSTM (128)
→ LSTM (64)
→ Dense (128)
→ Softmax (7 classes)
```

Optimizer: Adam (lr = 0.001)  
Loss: sparse_categorical_crossentropy  

---

## 📈 Results

Overall Accuracy: **91%**

The CNN-BiLSTM model significantly outperforms classical ML approaches such as SVM, Random Forest, and XGBoost.

---

## 🌐 Web Application

Run locally:

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📂 Reproducing the Project

1️⃣ Place raw audio files inside:

```
data/raw/
```

2️⃣ Run feature extraction notebook:

```
notebooks/02_feature_extraction.ipynb
```

This generates:

```
data/processed/features_gpu_final.csv
```

3️⃣ Run training notebook:

```
notebooks/03_model_training.ipynb
```

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
speech-emotion-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
|   ├── data.ipynb
│   ├── extration_caracteristique.ipynb
│   └── model.ipynb
│
├── models/
│   ├── my_model_gpu91.keras
│   ├── label_encoder.pkl
│   └── scaler2.pkl
│
├── data/
│   └── (pour les futures données)
│
└── pfe_we/
    ├── app.py
    ├── i.py
    ├── my_model_gpu91.keras
    ├── label_encoder.pkl
    ├── scaler2.pkl
    ├── saved_audio/
    ├── uploads/
    │   ├── angry0.wav
    │   └── suprise.wav
    ├── static/
    │   └── css/
    │       └── style.css
    └── templates/
        └── index.html
```

---

## 👤 Author

**Ilyass AIT CHEIKH**  
AI Developer | Deep Learning Enthusiast  
