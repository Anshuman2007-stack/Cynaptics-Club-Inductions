# Cynaptics-Club-Inductions
This is a repo containing the code for the Cynaptics Club Induction which contains a readme file and code. Also,I could only complete task 1 due to time constraint.
# 🎧 Audio Classification using CNN

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning project for **environmental sound classification** using
**Log-Mel Spectrograms** and a **Convolutional Neural Network (CNN)**.\
Optimized for simplicity, reproducibility, and deployment.

------------------------------------------------------------------------

# 📚 Table of Contents

1.  [Overview](#-overview)\
2.  [Features](#-features)\
3.  [Dataset Structure](#-dataset-structure)\
4.  [Model Architecture](#-model-architecture)\
5.  [Installation](#-installation)\
6.  [Training](#-training)\
7.  [Inference Example](#-inference-example)\
8.  [Future Improvements](#-future-improvements)\
9.  [License](#-license)

------------------------------------------------------------------------

# 🚀 Overview

This model classifies audio into **five categories**:

-   🐶 Dog Bark\
-   🔩 Drilling\
-   🚗 Engine Idling\
-   🚨 Siren\
-   🎵 Street Music

The approach uses:

-   **Mel-Spectrogram extraction**
-   **CNN-based feature learning**
-   **tf.data** for fast input pipelines\
-   **Early stopping** to avoid overfitting

------------------------------------------------------------------------

# ✨ Features

-   🎵 Convert WAV audio → Log-Mel Spectrogram\
-   ⚡ Fast training with `tf.data`\
-   🧠 CNN with BatchNorm + Dropout\
-   💾 Save/Load model using `.keras`\
-   📈 Includes validation tracking\
-   🔧 Highly modular and extendable

------------------------------------------------------------------------

# 📁 Dataset Structure

    dataset/
     ├── dog_bark/
     ├── drilling/
     ├── engine_idling/
     ├── siren/
     └── street_music/

Each folder contains `.wav` audio samples.

------------------------------------------------------------------------

# 🧠 Model Architecture

    Input: (400 × 64 × 1) Log-Mel Spectrogram
    │
    ├── Conv2D(32) → BatchNorm → MaxPool → Dropout
    ├── Conv2D(64) → BatchNorm → MaxPool → Dropout
    ├── Conv2D(128) → BatchNorm → MaxPool → Dropout
    │
    ├── Flatten
    ├── Dense(256, ReLU) → Dropout(0.5)
    └── Dense(5, Softmax)

------------------------------------------------------------------------

# 🛠 Installation

Install everything:

``` bash
pip install tensorflow keras librosa scikit-learn numpy matplotlib
```

------------------------------------------------------------------------

# 🎯 Training

Run the training script:

``` bash
python audio_classifier.py
```

The model is saved automatically at:

    C:\Python\Audio Classification_3.keras

------------------------------------------------------------------------

# 🔍 Inference Example

``` python
from keras.models import load_model
import numpy as np
import librosa

model = load_model(r"C:\\Python\\Audio Classification_3.keras")

def predict_audio(path):
    wav, sr = librosa.load(path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=64)
    logmel = librosa.power_to_db(mel).T
    logmel = logmel[:400, :]
    logmel = np.pad(logmel, ((0, max(0, 400 - logmel.shape[0])), (0, 0)))
    logmel = logmel[..., np.newaxis]
    pred = np.argmax(model.predict(logmel[np.newaxis]))
    return pred

print(predict_audio("test.wav"))
```

------------------------------------------------------------------------

# 🚧 Future Improvements

-   🔊 Audio Augmentation (noise, shift, stretch)\
-   🎼 CRNN (CNN + LSTM)\
-   🤖 Pretrained models like YAMNet, PANNs\
-   📱 Export to **TensorFlow Lite**\
-   🧪 Add test set accuracy reports

------------------------------------------------------------------------

# 📜 License

This project is released under the **MIT License**.
