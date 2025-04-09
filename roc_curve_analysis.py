import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42

def load_data():
    """Load the final features dataset"""
    try:
        # Load the final features
        X = pd.read_csv("final_features.csv")
        y = pd.read_csv("preprocessed_target.csv")
        
        # Convert y to a Series if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
            
        print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def prepare_data(X, y):
    """Prepare data for model training"""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def create_top_models():
    """Create a dictionary of the top performing models"""
    models = {
        # Top 3 models from previous evaluation
        'CatBoost': CatBoostClassifier(
            iterations=250, learning_rate=0.03, depth=14, min_data_in_leaf=5,
            subsample=0.85, l2_leaf_reg=3, class_weights=[1, 4], random_state=SEED, verbose=0
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            learning_rate=0.03, n_estimators=250, max_depth=10, min_samples_split=3,
            min_samples_leaf=2, subsample=0.85, random_state=SEED
        ),
        
        'Extra Trees Classifier': ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_split=4, random_state=SEED
        ),
        
        # Additional top models
        'XGBoost': XGBClassifier(
            colsample_bytree=0.8, learning_rate=0.03, max_depth=12, min_child_weight=4,
            n_estimators=250, subsample=0.85, gamma=0.1, reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=1, random_state=SEED
        ),
        
        'LightGBM': lgb.LGBMClassifier(
            colsample_bytree=0.9, learning_rate=0.03, max_depth=20, min_child_samples=5,
            n_estimators=250, num_leaves=90, subsample=0.85, class_weight='balanced',
            verbose=-1, random_state=SEED
        )
    }
    
    return models

def plot_roc_curves(models, X_train, X_test, y_train, y_test):
    """Plot ROC curves and calculate AUC for the best models"""
    # Create a figure for the ROC curves
    plt.figure(figsize=(12, 10))
    
    # Dictionary to store AUC scores
    auc_scores = {}
    
    # Colors for the ROC curves
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Train and evaluate each model
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name} for ROC curve analysis...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Get predicted probabilities for the positive class
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            # For models that don't have predict_proba
            y_score = model.decision_function(X_test)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = roc_auc
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()
    
    return auc_scores

def plot_precision_recall_curves(models, X_train, X_test, y_train, y_test):
    """Plot Precision-Recall curves for the best models"""
    # Create a figure for the Precision-Recall curves
    plt.figure(figsize=(12, 10))
    
    # Dictionary to store Average Precision scores
    ap_scores = {}
    
    # Colors for the Precision-Recall curves
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Train and evaluate each model
    for i, (name, model) in enumerate(models.items()):
        print(f"Calculating Precision-Recall curve for {name}...")
        
        # Get predicted probabilities for the positive class
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            # For models that don't have predict_proba
            y_score = model.decision_function(X_test)
        
        # Calculate Precision-Recall curve and Average Precision
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = average_precision_score(y_test, y_score)
        ap_scores[name] = average_precision
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, color=colors[i], lw=2,
                 label=f'{name} (AP = {average_precision:.3f})')
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png')
    plt.close()
    
    return ap_scores

def create_threshold_analysis(models, X_train, X_test, y_train, y_test):
    """Create threshold analysis for the best model"""
    # Get the best model (CatBoost based on previous evaluation)
    best_model = models['CatBoost']
    
    # Get predicted probabilities for the positive class
    y_score = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics at different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions based on threshold
        y_pred = (y_score >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        
        # Calculate sensitivity, specificity, and accuracy
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Calculate precision and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        results.append({
            'Threshold': threshold,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Accuracy': accuracy,
            'Precision': precision,
            'F1 Score': f1
        })
    
    # Create a DataFrame from results
    threshold_df = pd.DataFrame(results)
    
    # Save to CSV
    threshold_df.to_csv('threshold_analysis.csv', index=False)
    
    # Create a line plot for threshold analysis
    plt.figure(figsize=(12, 8))
    
    # Plot metrics vs threshold
    plt.plot(threshold_df['Threshold'], threshold_df['Sensitivity'], 'b-', label='Sensitivity')
    plt.plot(threshold_df['Threshold'], threshold_df['Specificity'], 'r-', label='Specificity')
    plt.plot(threshold_df['Threshold'], threshold_df['Accuracy'], 'g-', label='Accuracy')
    plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 'y-', label='Precision')
    plt.plot(threshold_df['Threshold'], threshold_df['F1 Score'], 'm-', label='F1 Score')
    
    # Add a vertical line at the default threshold (0.5)
    plt.axvline(x=0.5, color='k', linestyle='--', label='Default Threshold (0.5)')
    
    # Set plot properties
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.title('Threshold Analysis for CatBoost Model', fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('threshold_analysis.png')
    plt.close()
    
    return threshold_df

def main():
    # Load data
    X, y = load_data()
    if X is None or y is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Create top models
    models = create_top_models()
    
    # Plot ROC curves and calculate AUC
    auc_scores = plot_roc_curves(models, X_train, X_test, y_train, y_test)
    
    # Print AUC scores
    print("\nAUC Scores:")
    for name, score in auc_scores.items():
        print(f"{name}: {score:.4f}")
    
    # Plot Precision-Recall curves
    ap_scores = plot_precision_recall_curves(models, X_train, X_test, y_train, y_test)
    
    # Print Average Precision scores
    print("\nAverage Precision Scores:")
    for name, score in ap_scores.items():
        print(f"{name}: {score:.4f}")
    
    # Create threshold analysis for the best model
    threshold_df = create_threshold_analysis(models, X_train, X_test, y_train, y_test)
    
    # Find the optimal threshold based on F1 score
    optimal_threshold = threshold_df.loc[threshold_df['F1 Score'].idxmax(), 'Threshold']
    max_f1 = threshold_df['F1 Score'].max()
    
    print(f"\nOptimal Threshold (based on F1 Score): {optimal_threshold:.2f} (F1 Score: {max_f1:.4f})")
    
    print("\nROC curve analysis completed. Results saved to files.")

if __name__ == "__main__":
    main()
