# ZooVision AI- Automated Animal Classification

A dual-architecture machine learning web application that classifies animal species. It utilizes a custom-trained transfer learning model for primary classification and intelligently routes Out-of-Distribution (OOD) data to a global ImageNet fallback model.


## 🚀 Features

* **Dual-Brain Architecture:** 
  * **Tier 1 (Custom Model):** A lightweight `TensorFlow Lite` model fine-tuned on 15 specific animal classes using **MobileNetV2** transfer learning.
  * **Tier 2 (Global Fallback):** A full 1,000-class ImageNet model that activates only when the Tier 1 model detects anomalies or drops below confidence thresholds.
* **Out-of-Distribution (OOD) Detection:** Implements strict Softmax confidence limits (95%+) and decisiveness margins (40%+) to prevent neural network hallucinations (e.g., confidently misclassifying a turtle as a dog).
* **Modern Web Interface:** Built with **Streamlit**, featuring a side-by-side data analytics dashboard, probability distribution bar charts, and real-time WebRTC camera scanning.


## 🛠️ Technology Stack

* **Language:** Python 3.11+
* **Deep Learning:** TensorFlow, Keras, TFLite
* **Data Manipulation:** NumPy, Pandas
* **Frontend UI:** Streamlit
* **Computer Vision:** Pillow (PIL)


## Author
Kamal P
GitHub: https://github.com/15-Kamal
