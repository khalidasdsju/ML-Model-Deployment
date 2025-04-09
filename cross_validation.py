import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, make_scorer
)
from xgboost import XGBClassifier
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

def create_best_model():
    """Create the best model (XGBoost) with optimized hyperparameters"""
    try:
        # Try to load the best model
        model = joblib.load("best_xgboost_model.pkl")
        print("Loaded best XGBoost model from file")
    except:
        # Create a new model with the best parameters
        print("Creating new XGBoost model with optimized parameters")
        model = XGBClassifier(
            learning_rate=0.03,
            n_estimators=250,
            max_depth=12,
            min_child_weight=4,
            gamma=0.1,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    return model

def perform_cross_validation(model, X, y, n_folds=10):
    """Perform k-fold cross-validation"""
    # Define the cross-validation strategy
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    
    # Perform cross-validation
    print(f"Performing {n_folds}-fold cross-validation...")
    cv_results = cross_validate(
        model, X, y, 
        cv=cv, 
        scoring=scoring,
        return_train_score=True,
        return_estimator=True
    )
    
    # Extract results
    accuracy_scores = cv_results['test_accuracy']
    precision_scores = cv_results['test_precision']
    recall_scores = cv_results['test_recall']
    f1_scores = cv_results['test_f1']
    roc_auc_scores = cv_results['test_roc_auc']
    
    # Print results
    print("\nCross-Validation Results:")
    print(f"Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"ROC AUC: {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}")
    
    return cv_results

def visualize_cv_results(cv_results):
    """Visualize cross-validation results"""
    # Extract metrics
    metrics = {
        'Accuracy': cv_results['test_accuracy'],
        'Precision': cv_results['test_precision'],
        'Recall': cv_results['test_recall'],
        'F1 Score': cv_results['test_f1'],
        'ROC AUC': cv_results['test_roc_auc']
    }
    
    # Create a DataFrame for easier plotting
    results_df = pd.DataFrame(metrics)
    
    # Calculate statistics
    stats_df = pd.DataFrame({
        'Mean': results_df.mean(),
        'Std': results_df.std(),
        'Min': results_df.min(),
        'Max': results_df.max()
    })
    
    # Save statistics to CSV
    stats_df.to_csv('cv_statistics.csv')
    
    # Create a box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=results_df)
    plt.title('XGBoost 10-Fold Cross-Validation Results', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add mean value labels
    for i, metric in enumerate(metrics.keys()):
        plt.text(i, results_df[metric].mean() + 0.01, 
                 f'Mean: {results_df[metric].mean():.3f}', 
                 ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cv_results_boxplot.png')
    plt.close()
    
    # Create a violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df)
    plt.title('Distribution of Scores Across Folds', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cv_results_violinplot.png')
    plt.close()
    
    # Create a line plot showing scores across folds
    plt.figure(figsize=(12, 8))
    
    for metric in metrics.keys():
        plt.plot(range(1, len(metrics[metric]) + 1), metrics[metric], 'o-', label=metric)
    
    plt.title('Performance Across Folds', fontsize=16)
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(range(1, len(metrics['Accuracy']) + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cv_results_lineplot.png')
    plt.close()

def analyze_feature_importance(cv_results, X):
    """Analyze feature importance across folds"""
    # Extract feature importances from each fold
    feature_importances = []
    
    for estimator in cv_results['estimator']:
        feature_importances.append(estimator.feature_importances_)
    
    # Convert to DataFrame
    importance_df = pd.DataFrame(feature_importances, columns=X.columns)
    
    # Calculate mean and std of feature importances
    mean_importance = importance_df.mean()
    std_importance = importance_df.std()
    
    # Sort features by importance
    sorted_idx = mean_importance.argsort()[::-1]
    sorted_features = X.columns[sorted_idx]
    
    # Create a DataFrame for visualization
    importance_stats = pd.DataFrame({
        'Feature': sorted_features,
        'Mean Importance': mean_importance[sorted_idx],
        'Std Importance': std_importance[sorted_idx]
    })
    
    # Save to CSV
    importance_stats.to_csv('feature_importance_cv.csv', index=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 10))
    
    # Plot bars
    bars = plt.barh(
        range(len(sorted_features)), 
        mean_importance[sorted_idx],
        xerr=std_importance[sorted_idx],
        align='center',
        alpha=0.8
    )
    
    # Add feature names
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.title('Feature Importance (Mean ± Std) Across CV Folds', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + std_importance[sorted_idx][i] + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{mean_importance[sorted_idx][i]:.3f} ± {std_importance[sorted_idx][i]:.3f}',
            va='center',
            fontsize=9
        )
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_importance_cv.png')
    plt.close()
    
    return importance_stats

def main():
    # Load data
    X, y = load_data()
    if X is None or y is None:
        return
    
    # Create best model
    model = create_best_model()
    
    # Perform cross-validation
    cv_results = perform_cross_validation(model, X, y, n_folds=10)
    
    # Visualize cross-validation results
    visualize_cv_results(cv_results)
    
    # Analyze feature importance
    importance_stats = analyze_feature_importance(cv_results, X)
    
    print("\nTop 5 Most Important Features:")
    print(importance_stats.head(5))
    
    print("\nCross-validation analysis completed. Results saved to files.")

if __name__ == "__main__":
    main()
