# This Script is used to load data for the Breast Cancer Wisconsin Diagnostic Dataset


### LOADING DATA FOR BREAST CANCER DIAGNOSTIC DATASET- DATA COLLECTION
import pandas as pd

# Load the breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print("Data loaded successfully.")
# Saving the dataset into a CSV file
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('data/breast_cancer_data.csv', index= False)
print("Data saved to breast_cancer_data.csv successfully.")


### DATA UNDERSTANDING
# Convert the data to a pandas df
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target
print("Shape of the dataset: ", df.shape)
print("Data types: ", df.dtypes)
print("Target variable distribution: ", df['target'].value_counts())
print("Feature names: ", data.feature_names)
print("The Dataset has 30 features and 1 target variable.")
print("First few rows of the Breast Cancer Dataset: ", df.head())
print("Data understanding completed successfully.")

