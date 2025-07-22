# KNN Supervised Learning1
# 🧪 Breast Cancer Diagnostic Classification with KNN

This repository contains a practical project to build and evaluate a **K-Nearest Neighbors (KNN)** classification model to diagnose breast cancer from real patient data.  
Rather than using a quick notebook demo, this project is structured to reflect a **production-ready workflow** — focusing on clean, modular, and deployable Python code.

NB:
📖 If you’re not a coder and want to understand how and why key decisions were made in this project, please read my reflections and reasoning documented in the reports/ folder, and explore the visual summaries in plots_and_insights/.


MY AIM is to: 🎯
To design and implement a robust, modular, and production-ready machine learning pipeline that uses the k-Nearest Neighbors (KNN) algorithm to accurately classify breast cancer tumors as benign or malignant based on diagnostic measurements.

This project goes beyond exploratory notebook analysis by building a deployable Python-based ML system, reflecting real-world practices in data preprocessing, hyperparameter tuning, model evaluation, and model serialization — aligned with my learnings from prerecorded online clsses and theoretical knowledge

---

## 🎓 Background & Motivation

This project is part of my learning journey through the  
📚 **IBM Machine Learning Professional Certificate** on Coursera.  
Specifically:
- **Course:** 3 of 6 — *Supervised Machine Learning: Classification*
- **Module 2:** K-Nearest Neighbors

I’m using this hands-on project to:
- Apply the theory of KNN in a real-world medical dataset.
- Practice designing a machine learning pipeline for scalability and deployment.
- Explore hyperparameter tuning, evaluation, and feature preprocessing.

---

## 🩺 Dataset

- **Name:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** [`sklearn.datasets`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Description:** Contains 569 instances of benign and malignant tumors described by 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses.

This dataset is well-suited to demonstrate:
- The importance of feature scaling in distance-based algorithms.
- Hyperparameter tuning (e.g., choosing the best *k*).
- Trade-offs between model complexity, bias, and variance.

---

## ⚙️ Project Structure

```text
.
├── data/                  # Data loading or download scripts if needed
├── src/                   # Source code for data processing,         training, and evaluation
│   ├── preprocess.py      # Feature scaling, train-test split
│   ├── train.py          # Model training and hyperparameter tuning
│   ├── evaluate.py      # Evaluation metrics, confusion matrix
│   └── utils.py         # Helper functions
├── models/                # Saved trained models (pickle or joblib)
├── requirements.txt      # Python dependencies
└── README.md



