import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    """Load preprocessed data"""
    try:
        # Load preprocessed features and target
        X = pd.read_csv("preprocessed_features.csv")
        y = pd.read_csv("preprocessed_target.csv")

        # Convert y to a Series if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # Check for non-numeric columns and convert them
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"Converting column {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')

                # Fill NaN values with the mean or median
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())

        # Check for any remaining non-numeric columns
        non_numeric_cols = [col for col in X.columns if X[col].dtype == 'object']
        if non_numeric_cols:
            print(f"Warning: The following columns are still non-numeric: {non_numeric_cols}")
            # Drop these columns as a last resort
            X = X.drop(columns=non_numeric_cols)
            print(f"Dropped non-numeric columns. New X shape: {X.shape}")

        print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_lightgbm_model(X, y):
    """Train a LightGBM model"""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # Define LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Train the model
    print("Training LightGBM model...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=callbacks
    )

    # Make predictions
    y_pred = np.round(model.predict(X_test))

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return model, X_train, X_test, y_train, y_test

def calculate_shap_values(model, X_test):
    """Calculate SHAP values for feature importance"""
    print("Calculating SHAP values...")

    # Get feature names directly from the model
    feature_names = model.feature_name()

    # Create a DataFrame for SHAP values
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Create the SHAP explainer
    explainer = shap.Explainer(model)

    # Calculate SHAP values
    shap_values = explainer(X_test_df)

    return shap_values, X_test_df, feature_names

def visualize_feature_importance(shap_values, X_test_df, feature_names):
    """Visualize feature importance using SHAP values"""
    # Summary plot
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP Values)")
    plt.tight_layout()
    plt.savefig("feature_importance_bar.png")
    plt.close()

    # Summary plot with all data points
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("Feature Importance Summary")
    plt.tight_layout()
    plt.savefig("feature_importance_summary.png")
    plt.close()

    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(0)

    # Create a DataFrame with feature names and importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # Save to CSV
    feature_importance.to_csv("feature_importance.csv", index=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))

    return feature_importance

def identify_best_features(feature_importance, threshold=0.9):
    """Identify the best features based on cumulative importance"""
    # Calculate cumulative importance
    feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum() / feature_importance['Importance'].sum()

    # Find features that contribute to threshold of importance
    best_features = feature_importance[feature_importance['Cumulative_Importance'] <= threshold]

    print(f"\nBest Features (contributing to {threshold*100}% of importance):")
    print(best_features)

    # Save best features to CSV
    best_features.to_csv("best_features.csv", index=False)

    return best_features['Feature'].tolist()

def main():
    # Load data
    X, y = load_data()
    if X is None or y is None:
        return

    # Train LightGBM model
    model, X_train, X_test, y_train, y_test = train_lightgbm_model(X, y)

    # Calculate SHAP values
    shap_values, X_test_df, feature_names = calculate_shap_values(model, X_test)

    # Visualize feature importance
    feature_importance = visualize_feature_importance(shap_values, X_test_df, feature_names)

    # Identify best features
    best_features = identify_best_features(feature_importance)

    print(f"\nNumber of best features selected: {len(best_features)}")

    # Create a reduced dataset with only the best features
    X_reduced = X[best_features]
    X_reduced.to_csv("reduced_features.csv", index=False)

    print("\nReduced feature dataset saved to 'reduced_features.csv'")

if __name__ == "__main__":
    main()
