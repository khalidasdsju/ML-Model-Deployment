import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load preprocessed data"""
    try:
        # Load preprocessed features and target
        X = pd.read_csv("preprocessed_features.csv")
        y = pd.read_csv("preprocessed_target.csv")
        
        # Convert y to a Series if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
            
        print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def load_shap_features():
    """Load SHAP-selected features"""
    try:
        # Load the best features identified by SHAP
        shap_features = pd.read_csv("best_features.csv")
        print(f"Loaded {len(shap_features)} SHAP-selected features")
        return shap_features['Feature'].tolist()
    except Exception as e:
        print(f"Error loading SHAP features: {e}")
        return []

def apply_domain_expert_selection(X, shap_features):
    """Apply domain expert feature selection"""
    # List of features to drop based on domain expert knowledge
    features_to_drop = ["BA", "HbA1C", "Na", "K", "Cl", "Hb", "MPI", "HDLc"]
    
    print("\nFeatures to drop based on domain expert knowledge:")
    print(features_to_drop)
    
    # Check which features from domain expert list are in the dataset
    available_features_to_drop = [f for f in features_to_drop if f in X.columns]
    
    if len(available_features_to_drop) < len(features_to_drop):
        missing = set(features_to_drop) - set(available_features_to_drop)
        print(f"\nWarning: Some domain expert features not found in dataset: {missing}")
    
    # Check which features to drop are also in SHAP-selected features
    overlap = set(available_features_to_drop).intersection(set(shap_features))
    if overlap:
        print(f"\nNote: The following features are both in domain expert drop list and SHAP important features:")
        print(list(overlap))
    
    # Drop the features
    X_reduced = X.drop(columns=available_features_to_drop)
    print(f"\nRemoved {len(available_features_to_drop)} features based on domain expert knowledge")
    print(f"New dataset shape: {X_reduced.shape}")
    
    return X_reduced, available_features_to_drop

def combine_feature_selection_approaches(X, shap_features):
    """Combine SHAP and domain expert feature selection"""
    # List of features to drop based on domain expert knowledge
    features_to_drop = ["BA", "HbA1C", "Na", "K", "Cl", "Hb", "MPI", "HDLc"]
    
    # Keep only SHAP features that are not in the domain expert drop list
    final_features = [f for f in shap_features if f not in features_to_drop]
    
    print(f"\nFinal feature count after combining approaches: {len(final_features)}")
    print("Final features:")
    print(final_features)
    
    # Select only these features from the dataset
    X_final = X[final_features]
    
    return X_final, final_features

def visualize_feature_comparison(shap_features, domain_expert_drops, final_features):
    """Visualize the feature selection comparison"""
    # Create sets for comparison
    shap_set = set(shap_features)
    domain_drop_set = set(domain_expert_drops)
    final_set = set(final_features)
    
    # Create a Venn diagram-like visualization
    plt.figure(figsize=(12, 8))
    
    # Bar chart showing feature counts
    counts = [
        len(shap_set), 
        len(domain_drop_set), 
        len(shap_set.intersection(domain_drop_set)),
        len(final_set)
    ]
    
    labels = [
        'SHAP Selected', 
        'Domain Expert Drops', 
        'Overlap', 
        'Final Features'
    ]
    
    plt.bar(labels, counts, color=['blue', 'red', 'purple', 'green'])
    plt.title('Feature Selection Comparison', fontsize=16)
    plt.ylabel('Number of Features', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png')
    plt.close()

def main():
    # Load data
    X, y = load_data()
    if X is None:
        return
    
    # Load SHAP-selected features
    shap_features = load_shap_features()
    if not shap_features:
        return
    
    # Apply domain expert selection
    X_domain_reduced, domain_expert_drops = apply_domain_expert_selection(X, shap_features)
    
    # Combine both approaches
    X_final, final_features = combine_feature_selection_approaches(X, shap_features)
    
    # Visualize the comparison
    visualize_feature_comparison(shap_features, domain_expert_drops, final_features)
    
    # Save the final dataset
    X_final.to_csv("final_features.csv", index=False)
    
    print("\nFinal feature dataset saved to 'final_features.csv'")
    
    # Create a feature importance summary
    feature_summary = pd.DataFrame({
        'Feature': list(set(shap_features + domain_expert_drops)),
        'In_SHAP_Selection': [f in shap_features for f in set(shap_features + domain_expert_drops)],
        'In_Domain_Expert_Drops': [f in domain_expert_drops for f in set(shap_features + domain_expert_drops)],
        'In_Final_Selection': [f in final_features for f in set(shap_features + domain_expert_drops)]
    })
    
    feature_summary.to_csv("feature_selection_summary.csv", index=False)
    print("Feature selection summary saved to 'feature_selection_summary.csv'")

if __name__ == "__main__":
    main()
