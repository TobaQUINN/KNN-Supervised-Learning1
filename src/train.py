# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os

# I will be writing this in function modular design, so i'll just simply call them at the end, easier

# Function to load the dataset
def load_data(path='data/breast_cancer_preprocessed_data.csv'):
    df = pd.read_csv(path)
    return df

# Function to split the preprocessed data into training and validation sets
def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['target']) # Features
    y = df['target']  # Target variable to be predicted, the test split does not have this, the training set is trained with the target
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Function to tune k using cross validation. Returns best k and the list of mean accuracies
def tune_k(X_train, y_train, k_range=range(1, 21), cv=5):
    mean_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        mean_scores.append(scores.mean())
    best_k = k_range[np.argmax(mean_scores)]
    return best_k, mean_scores

# Functin to train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]  # for ROC AUC

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=['malignant', 'benign'])

    print("\n=== Test set evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print("\nClassification report:\n", report)

    return model, {'accuracy': acc, 'f1_score': f1, 'roc_auc': roc, 'report': report}

# Function to save the model using joblib
def save_model(model, path='models/breast_cancer_predictor.joblib'):  # The path already exists
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

# Function to plot k vs cross valdated accuracy
def plot_k_vs_accuracy(k_range, mean_scores, output_path='plots/k_vs_accuracy.png'):
    
    plt.figure(figsize=(8,5))
    plt.plot(k_range, mean_scores, marker='o')
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Cross-validated accuracy')
    plt.title('KNN model selection')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig('plots_and_insights/output_path')
    plt.close()
    print(f"📊 Plot saved to {output_path}")

# Function to carry out the overall training and evaluation process from other functions, a function that calls other functions :)
def main():
    print(" Loading data...")
    df = load_data()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Tuning k...")
    k_range = range(1, 21)
    best_k, mean_scores = tune_k(X_train, y_train, k_range)

    print(f"Best k found: {best_k}")

    print("Plotting k vs accuracy...")
    plot_k_vs_accuracy(k_range, mean_scores)

    print("Training final model...")
    model, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, best_k)

    print("Saving model as breast_cancer_predictor...")
    save_model(model)

    # Save metrics report with UTF-8 encoding
    with open('reports/model_report.md', 'w', encoding='utf-8') as f:
        f.write(f"# 🧪 Model Metrics\n\n")
        f.write(f"- **Best k (chosen by cross-validation)**: {best_k}\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
        f.write(f"- **F1 score**: {metrics['f1_score']:.4f}\n")
        f.write(f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n\n")
        f.write(f"## Classification Report\n\n")
        f.write(f"```\n{metrics['report']}\n```\n")
        f.write(f"\n![k vs Accuracy](../plots/k_vs_accuracy.png)\n")

if __name__ == "__main__":
    main()
