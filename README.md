# 🎤 Reconnaissance des Émotions Vocales avec CNN-BiLSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green)](https://librosa.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🎓 Projet de Fin d'Études – BTS Développement en Intelligence Artificielle
> 👨‍💻 Auteur : **Ilyass AIT CHEIKH**
> 📅 Année académique : 2024–2025

---

## 🚀 Présentation du Projet

Ce projet présente un **système de Deep Learning pour la Reconnaissance des Émotions Vocales (SER)** basé sur une architecture hybride **CNN-BiLSTM**.

Le système classifie les signaux vocaux en **7 émotions** :

- 😠 Colère
- 🤢 Dégoût
- 😨 Peur
- 😊 Joie
- 😐 Neutre
- 😢 Tristesse
- 😲 Surprise

### 🎯 Performance Finale
- ✅ Précision : **91%**
- ✅ Score F1 macro : **0.91**
- ✅ Bonne généralisation sur plusieurs datasets

Une **application web Flask** est incluse pour la prédiction d'émotions en temps réel.

---

## 📊 Jeux de Données

Le modèle a été entraîné sur quatre datasets publics :

| Dataset | Description |
|---------|-------------|
| CREMA-D | 7 442 clips audio de 91 acteurs |
| RAVDESS | Discours et chant avec 24 acteurs |
| TESS | 2 800 stimuli audio de femmes âgées |
| SAVEE | 480 utterances de 4 locuteurs masculins |

> ⚠️ Les fichiers audio bruts ne sont **PAS inclus** dans ce dépôt en raison de leur taille.

---

## 🔧 Pipeline de Traitement

### 1️⃣ Prétraitement Audio
- Durée fixe : 2,5 secondes
- Suppression des silences
- Normalisation de l'amplitude

### 2️⃣ Augmentation des Données
- Bruit gaussien
- Étirement temporel
- Décalage de hauteur tonale
- Décalage temporel
- Mise à l'échelle du volume
- Perturbation de la vitesse

### 3️⃣ Extraction de Caractéristiques
- 40 MFCC
- Spectrogramme Mel
- ZCR (Zero Crossing Rate)
- Énergie RMS
- Chroma STFT
- Contraste Spectral
- Tonnetz

### 4️⃣ Équilibrage des Classes
- SMOTE

### 5️⃣ Mise à l'Échelle
- StandardScaler

---

## 🧠 Architecture du Modèle
```
Entrée (195, 1)
→ Conv1D (128) + BatchNorm + Pooling + Dropout
→ Conv1D (256) + Pooling + Dropout
→ Conv1D (512) + Pooling + Dropout
→ LSTM Bidirectionnel (128)
→ LSTM (64)
→ Dense (128)
→ Softmax (7 classes)
```

- **Optimiseur :** Adam (lr = 0.001)
- **Fonction de perte :** sparse_categorical_crossentropy

---

## 📈 Résultats

| Métrique | Score |
|----------|-------|
| Précision globale | **91%** |
| Score F1 macro | **0.91** |

> Le modèle CNN-BiLSTM surpasse largement les approches classiques : SVM, Random Forest et XGBoost.

---

## 🌐 Application Web

Lancer en local :
```bash
python app.py
```

Ouvrir dans le navigateur :
```
http://127.0.0.1:5000
```

---

## 📂 Reproduire le Projet

**1️⃣** Placer les fichiers audio bruts dans :
```
data/raw/
```

**2️⃣** Exécuter le notebook d'extraction de caractéristiques :
```
notebooks/extration_caracteristique.ipynb
```

Génère :
```
data/processed/features_gpu_final.csv
```

**3️⃣** Exécuter le notebook d'entraînement :
```
notebooks/model.ipynb
```

---

## 🛠️ Installation
```bash
git clone https://github.com/ton-username/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```

---

## 📁 Structure du Projet
```
speech-emotion-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── data.ipynb
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

## 👤 Auteur

**Ilyass AIT CHEIKH**
AI Developer | Passionné de Deep Learning

