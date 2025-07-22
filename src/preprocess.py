### EXPLORATORY DATA ANALYSIS & DATA PREPROCESSING

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Check for basic statistics
print("Basic statistics of the dataset: ", df.describe())

# Check for missing values
print("Missing values in the dataset: ", df.isnull().sum())
print("There are no missing values in the dataset.")

# Check for duplicates
print("Number of Duplicates in the dataset: ", df.duplicated().sum())
print("There are no duplicates in the dataset.")


# Check for outliers
# Here I check for outliers to ensure data qualiity, removal is not an option as this is a medical dataset, an outlier here can be a critical case
# Since we have so many features, I will visualize the distribution of each feature iteratively using boxplots

# Dropping the target column to leave only feature for outlier detection
features = df.columns.drop('target')

# One big figure with all boxplots
n_cols = 5
n_rows = (len(features) + n_cols - 1) // n_cols

plt.figure(figsize=(4 * n_cols, 2.5 * n_rows))

for i, col in enumerate(features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col, fontsize=9)
    plt.tight_layout()

plt.suptitle('Boxplots of all features', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('plots_and_insights/boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# IQR method
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

print("All the features have outliers")

# Now that i have visualized the outliers, I will not remove them as they are critical cases in a medical dataset.
# What do i Do? I will try to check if the outliers relates to the target being benign or malignant, by checking the target distribution of outliers and non-outliers for each feature.
# We have so many features, so i will visualize the target distribution of outliers and non outliers as subplots in a single feature

# I will define a helper function to get the outliers for each feature
def get_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[feature] < lower) | (df[feature] > upper)]

# Visualizing the target distribution of outliers for each feature
plt.figure(figsize=(4 * n_cols, 2.5 * n_rows))

for i, col in enumerate(features, 1):
    plt.subplot(n_rows, n_cols, i)
    outliers = get_outliers(df, col)
    if not outliers.empty:
        sns.countplot(x='target', data=outliers, palette='pastel')
        plt.title(f'{col} outliers (n={len(outliers)})', fontsize=8)
    else:
        plt.title(f'{col}: no outliers', fontsize=8)
    plt.tight_layout()

plt.suptitle('Target distribution among outliers by feature', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('plots_and_insights/target_distribution_outliers.png', dpi=300, bbox_inches='tight')
plt.show()

# Consider the fact: the mean of the target is the proportion of cases, Closer to 1 is benign, and closer to zero is malignant, according to data.target_names saved in the sklearn datasets
# I can see from the countplots that the features have outliers with target means closer to 0, indicating that the outliers are more likely to be malignant cases.
# Since I had no intention to remove the outliers, plus the fact that they are likely to be malignant cases, they are not noises, they would contribute to the model's ability to detect malignant cases.
# Categorical features here are meant to predict 1-benign, 0-malignant
# However we have features with outliers that are very important to predict malignant cases, but the fact that they are outliers won't change, so i have to find a way to optimize the model's performance with this and prevent the negative effect of the outliers.
# I will use scaling and hyperparameter tuning to limit their negative effect on distance calculations in train.py script.

## Checking for correlation to reduce redundancy of features, and improve model's performance.
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidth=1.5)
plt.title('Correlation Matrix of Breast Cancer Dataset 30 Features')
plt.tight_layout()
plt.savefig('plots_and_insights/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# Now that i have plotted the correlation matrix, I can see that some of the features are highly correlated, which will bias the distance calculations
# KNN is sensitive to Redundant features, so I will drop the highly correlated features to improve the model's performance.
# But my screen is small, and i should save time from counting and identifying correlated features, so I will use a helper function to drop the highly correlated features.


import pandas as pd

def drop_highly_correlated_features(df, target_col='target', corr_threshold=0.9, verbose=True):

    # Compute correlation matrix (exclude target)
    corr_matrix = df.drop(columns=[target_col]).corr().abs()

    # Track features to drop
    to_drop = set()

    # Compute target correlation for each feature
    target_corr = df.drop(columns=[target_col]).corrwith(df[target_col]).abs()

    # Iterate over the upper triangle of the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            if corr_value > corr_threshold:
                # Compare absolute correlation with target
                if target_corr[feature1] >= target_corr[feature2]:
                    drop = feature2
                else:
                    drop = feature1

                if verbose:
                    print(f"Dropping '{drop}' (corr={corr_value:.2f}) "
                          f"because it is highly correlated with '{feature1 if drop==feature2 else feature2}' "
                          f"and less predictive (|corr with target| {target_corr[drop]:.2f})")

                to_drop.add(drop)

    # Drop selected features
    reduced_df = df.drop(columns=list(to_drop))

    if verbose:
        print(f"\nTotal features dropped: {len(to_drop)}")
        print(f"Remaining features: {len(reduced_df.columns)-1} + target")

    return reduced_df

# This function wil drop the highly correlated features and keeps the feature more predictive of the target
# Now i will be using the drop function

reduced_df = drop_highly_correlated_features(df, target_col='target', corr_threshold=0.9)

# Save the reduced dataset to a new CSV file
reduced_df.to_csv('data/breast_cancer_reduced_data.csv', index=False)
print("Reduced dataset saved to breast_cancer_reduced_data.csv successfully.")

# Finally I will be scaling the features to prevent the outliers from negatively affecting the model's performance
from sklearn.preprocessing import StandardScaler
# Initialize the scaler
scaler = StandardScaler()
# Fit the scaler to the features and transform them
scaled_features = scaler.fit_transform(reduced_df.drop(columns=['target']))
# Create a new DataFrame with the scaled features and the target
scaled_df = pd.DataFrame(scaled_features, columns=reduced_df.columns.drop('target'))
scaled_df['target'] = reduced_df['target'].values
# Save the scaled dataset to a new CSV file
scaled_df.to_csv('data/breast_cancer_scaled_data.csv', index=False)
print("Scaled dataset saved to breast_cancer_scaled_data.csv successfully.")

# Saving the final file as preprocessed data
scaled_df.to_csv('data/breast_cancer_preprocessed_data.csv', index=False)
print("Preprocessed dataset saved to breast_cancer_preprocessed_data.csv successfully.")
