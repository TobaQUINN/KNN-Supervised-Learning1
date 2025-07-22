import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import joblib

def load_data(path='data/breast_cancer_preprocessed_data.csv'):
    return pd.read_csv(path)

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def tune_k(X_train, y_train, k_range=range(1, 21), cv=5):
    mean_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        mean_scores.append(scores.mean())
    best_k = k_range[np.argmax(mean_scores)]
    return best_k, mean_scores

def train_and_evaluate(X_train, y_train, X_test, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
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

def save_model(model, path='models/breast_cancer_predictor.joblib'):
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def plot_k_vs_accuracy(k_range, mean_scores, output_path='plots/k_vs_accuracy.png'):
    plt.figure(figsize=(8,5))
    plt.plot(k_range, mean_scores, marker='o')
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Cross-validated accuracy')
    plt.title('KNN model selection')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Plot saved to {output_path}")
