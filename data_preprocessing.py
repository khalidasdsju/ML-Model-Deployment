import numpy as np
import pandas as pd
from scipy.stats import skew, normaltest, stats
from sklearn.preprocessing import LabelEncoder, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Function to detect outliers using IQR method
def detect_outliers_iqr(df, threshold=1.5):
    outliers = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Find outliers
        outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        if len(outlier_indices) > 0:
            outliers[column] = len(outlier_indices)

    return outliers

# Function to apply capping (winsorization)
def cap_outliers(df, columns, lower_percentile=0.01, upper_percentile=0.99):
    df_capped = df.copy()
    for column in columns:
        lower_limit = df[column].quantile(lower_percentile)
        upper_limit = df[column].quantile(upper_percentile)
        df_capped[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
        df_capped[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
    return df_capped

# Function to perform encoding based on the recommended encoding method
def encode_columns(data):
    # Label Encoder initialization
    le = LabelEncoder()

    # Create a copy to avoid modifying the original dataframe
    encoded_data = data.copy()

    # Define the columns and their respective encoding methods
    label_encoding_columns = [
        'Sex', 'NYHA', 'HTN', 'DM', 'Smoker', 'DL', 'BA', 'CXR', 'RWMA', 'MI',
        'Chest_pain', 'HF'
    ]

    one_hot_encoding_columns = [
        'ECG', 'ACS', 'Wall', 'Thrombolysis'
    ]

    # Apply Label Encoding to binary categorical variables
    for col in label_encoding_columns:
        if col in encoded_data.columns:
            encoded_data[col] = le.fit_transform(encoded_data[col])

    # Apply One-Hot Encoding to nominal categorical variables
    encoded_data = pd.get_dummies(encoded_data, columns=one_hot_encoding_columns, drop_first=True)

    return encoded_data

# Function to visualize distributions before and after transformation
def plot_distributions(original, transformed, columns, rows=4, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(18, 16))
    axes = axes.flatten()

    for i, col in enumerate(columns[:rows*cols]):
        if col in original.columns and col in transformed.columns:
            # Original distribution
            sns.histplot(original[col].dropna(), ax=axes[i], color='blue', alpha=0.5, label='Original')

            # Transformed distribution
            sns.histplot(transformed[col].dropna(), ax=axes[i], color='red', alpha=0.5, label='Transformed')

            axes[i].set_title(f'Distribution of {col}')
            axes[i].legend()

    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    plt.close()

# Main preprocessing function
def preprocess_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Drop the StudyID column if it exists
    if 'StudyID' in data.columns:
        data = data.drop('StudyID', axis=1)

    # Create a copy for transformed data
    data_transformed = data.copy()

    # Initialize lists to store column classifications
    normal_dist = []
    right_skewed = []
    left_skewed = []
    non_normal_dist = []

    # Iterate over each numeric column in the dataset
    for column in data.select_dtypes(include=[np.number]).columns:
        # Calculate skewness
        feature_skewness = skew(data[column].dropna())  # Drop NaN values before calculation

        # Check for normality using D'Agostino's K-squared test (normality test)
        _, p_value = normaltest(data[column].dropna())  # p_value < 0.05 means the data is not normal

        if abs(feature_skewness) < 0.5 and p_value > 0.05:
            # If skewness is close to 0 and data passes the normality test
            normal_dist.append(column)
        elif feature_skewness > 0.5:
            # If skewness is positive (right skewed)
            right_skewed.append(column)
        elif feature_skewness < -0.5:
            # If skewness is negative (left skewed)
            left_skewed.append(column)

        # Classify as non-normal if the p-value is < 0.05 (fails normality test)
        if p_value < 0.05:
            non_normal_dist.append(column)

    # Print the classification results
    print("Normal Distribution:", normal_dist)
    print("Right Skewed:", right_skewed)
    print("Left Skewed:", left_skewed)
    print("Non-Normal Distribution:", non_normal_dist)

    # Check outliers before transformation
    outliers_before = detect_outliers_iqr(data)
    print("\nOutliers Before Transformation:")
    print(outliers_before)

    # 1. Apply Log Transformation to Right-Skewed Features
    right_skewed = ['BMI', 'HR', 'RBS', 'Creatinine', 'K', 'LVIDs', 'MPI', 'RR', 'TC', 'LDLc', 'HDLc', 'TG']

    for column in right_skewed:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')  # Ensure numeric values
            if data[column].notna().all() and (data[column] > 0).all():
                data_transformed[column] = np.log(data[column] + 1)

    # 2. Apply Square Root or Cube Root Transformation to Moderately Skewed Features
    moderately_skewed = ['HbA1C', 'Cl', 'TropI', 'BNP']  # Example for moderately skewed features

    for column in moderately_skewed:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')  # Ensure numeric values
            if data[column].notna().all() and (data[column] > 0).all():
                data_transformed[column] = np.sqrt(data[column] + 1)  # Square root transformation

    # 3. Apply Box-Cox Transformation to Non-Normal Distributed Features
    non_normal_dist = ['Na', 'K', 'Hb', 'LVIDd', 'FS', 'LVEF', 'LAV', 'IRT', 'EA', 'DT', 'RR']

    for column in non_normal_dist:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')  # Ensure numeric values
            if data[column].notna().all() and (data[column] > 0).all():
                try:
                    data_transformed[column], _ = stats.boxcox(data[column] + 1)  # Add 1 to avoid log(0)
                except:
                    print(f"Could not apply Box-Cox transformation to {column}")

    # Apply capping to all numerical columns
    numeric_cols = data_transformed.select_dtypes(include=[np.number]).columns
    data_transformed = cap_outliers(data_transformed, numeric_cols)

    # Check outliers after capping
    outliers_after_capping = detect_outliers_iqr(data_transformed)
    print("\nOutliers After Capping:")
    print(outliers_after_capping)

    # Apply encoding to categorical columns
    data_encoded = encode_columns(data_transformed)

    # Visualize distributions before and after transformation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    plot_distributions(data, data_transformed, numeric_cols)

    # Split the data into features and target
    X = data_encoded.drop('HF', axis=1) if 'HF' in data_encoded.columns else data_encoded
    y = data_encoded['HF'] if 'HF' in data_encoded.columns else None

    # Select only numeric columns for scaling
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]

    # Initialize RobustScaler
    scaler = RobustScaler()

    # Fit and transform only the numeric features
    X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_cols)

    # Combine scaled numeric features with categorical features
    X_scaled = X.copy()
    X_scaled[numeric_cols] = X_numeric_scaled

    print("\nPreprocessing completed successfully!")

    return X_scaled, y, data_transformed, data_encoded

if __name__ == "__main__":
    # Path to your dataset
    data_path = "artifact/04_07_2025_15_36_55/data_ingestion/data_ingestion/train.csv"

    # Preprocess the data
    X_scaled, y, data_transformed, data_encoded = preprocess_data(data_path)

    # Save the preprocessed data
    X_scaled.to_csv("preprocessed_features.csv", index=False)
    if y is not None:
        y.to_csv("preprocessed_target.csv", index=False)
    data_transformed.to_csv("transformed_data.csv", index=False)
    data_encoded.to_csv("encoded_data.csv", index=False)

    print("Preprocessed data saved to CSV files.")
