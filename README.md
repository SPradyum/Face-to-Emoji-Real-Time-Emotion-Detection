# 🎭 Face to Emoji – Real-Time Emotion Detection

A real-time facial emotion recognition application that uses a webcam feed to detect human emotions and display corresponding emojis. Built using **TensorFlow / Keras**, **OpenCV**, and a responsive **Tkinter GUI**.

📥 Download the dataset (archive) + pretrained model (model.h5) here:
👉 https://drive.google.com/drive/folders/1hBkzDQOaPHeGf1_iVIjoeg3yJEhy8TRr?usp=drive_link
---

## 🚀 Features

- **Real-time emotion detection** using webcam input  
- **Deep CNN model** trained on the FER2013 dataset  
- **Instant emoji display** based on detected emotion  
- **Optimized GUI** (no lag, low CPU usage)  
- **Preloaded emojis** for fast performance  
- **Compatible with Python 3.10–3.13 & Keras 3**  

---

## 📂 Project Structure

```
Face-to-Emoji/
│
├── train.py             # Model training script (FER2013 or custom dataset)
├── gui.py               # Real-time emotion detection + emoji display GUI
├── model.h5             # Saved trained model
├── emojis/              # Emoji images (PNG)
│   ├── angry.png
│   ├── disgusted.png
│   ├── fearful.png
│   ├── happy.png
│   ├── neutral.png
│   ├── sad.png
│   └── surprised.png
├── archive/             # Dataset folder (FER2013)
│   ├── train/
│   └── test/
└── README.md
```

---

## 🧠 Model Information

The model is a **Convolutional Neural Network (CNN)** trained to classify 7 emotions:

- Angry  
- Disgusted  
- Fearful  
- Happy  
- Neutral  
- Sad  
- Surprised  

Input shape: **48 × 48 × 1 (Grayscale)**  
Output: **7-class Softmax**  

---

## 📦 Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
```

Recommended versions:
```
tensorflow==2.15 or tensorflow-cpu==2.16
opencv-python==4.9+
pillow
numpy
```

---

## 🎓 Training the Model

If you want to retrain the model:

```bash
python train.py
```

This will:

- Load dataset from `archive/train` and `archive/test`
- Train the CNN for 50 epochs
- Save the final model as:
```
model.h5
```

---

## 🖥️ Running the GUI

Start the real-time emotion detector with:

```bash
python gui.py
```

The app will:

- Open your webcam  
- Detect your face  
- Predict emotion every 100ms  
- Display the matching emoji  

---

## 🧪 Dataset Format (FER2013)

Your dataset must follow this structure:

```
archive/
 ├── train/
 │    ├── angry/
 │    ├── disgusted/
 │    ├── fearful/
 │    ├── happy/
 │    ├── neutral/
 │    ├── sad/
 │    └── surprised/
 └── test/
      ├── angry/
      ├── disgusted/
      ├── fearful/
      ├── happy/
      ├── neutral/
      ├── sad/
      └── surprised/
```
---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Tkinter**
- **NumPy**
- **Pillow (PIL)**

---

## 🤝 Contributing

Pull requests are welcome!  
If you'd like new features (e.g., sound effects, animated emojis, probability bars), submit an issue.

---

## ⭐ Support

If you like this project, give it a **star** ⭐ on GitHub!
