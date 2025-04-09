import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
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

def load_best_models():
    """Load the best models from Optuna optimization"""
    try:
        # Load the best models
        xgboost_model = joblib.load("best_xgboost_model.pkl")
        lightgbm_model = joblib.load("best_lightgbm_model.pkl")
        catboost_model = joblib.load("best_catboost_model.pkl")
        
        print("Best models loaded successfully")
        return {
            'XGBoost': xgboost_model,
            'LightGBM': lightgbm_model,
            'CatBoost': catboost_model
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        
        # Create new models with the best parameters from the optimization
        print("Creating new models with the best parameters...")
        
        # XGBoost best parameters
        xgboost_model = XGBClassifier(
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
        
        # LightGBM best parameters
        lightgbm_model = lgb.LGBMClassifier(
            learning_rate=0.03,
            n_estimators=250,
            num_leaves=90,
            max_depth=20,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.9,
            class_weight='balanced',
            random_state=SEED,
            verbose=-1
        )
        
        # CatBoost best parameters
        catboost_model = CatBoostClassifier(
            learning_rate=0.03,
            iterations=250,
            depth=14,
            min_data_in_leaf=5,
            subsample=0.85,
            l2_leaf_reg=3,
            class_weights=[1, 4],
            random_seed=SEED,
            verbose=0
        )
        
        return {
            'XGBoost': xgboost_model,
            'LightGBM': lightgbm_model,
            'CatBoost': catboost_model
        }

def train_models(models, X_train, y_train):
    """Train the best models"""
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate the best models"""
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'F1 Score': f1,
            'Predictions': y_pred,
            'Probabilities': y_prob
        }
        
        # Print results
        print(f"{name} performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    return results

def create_ensemble_predictions(results, X_test, y_test):
    """Create ensemble predictions using voting and averaging"""
    # Get probabilities from each model
    probabilities = np.column_stack([
        results['XGBoost']['Probabilities'],
        results['LightGBM']['Probabilities'],
        results['CatBoost']['Probabilities']
    ])
    
    # Average probabilities (soft voting)
    avg_probabilities = np.mean(probabilities, axis=1)
    
    # Hard voting (majority vote)
    predictions = np.column_stack([
        results['XGBoost']['Predictions'],
        results['LightGBM']['Predictions'],
        results['CatBoost']['Predictions']
    ])
    hard_vote_predictions = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(), 
        axis=1, 
        arr=predictions
    )
    
    # Calculate metrics for soft voting
    soft_vote_predictions = (avg_probabilities >= 0.5).astype(int)
    soft_accuracy = accuracy_score(y_test, soft_vote_predictions)
    soft_roc_auc = roc_auc_score(y_test, avg_probabilities)
    soft_f1 = f1_score(y_test, soft_vote_predictions)
    
    # Calculate metrics for hard voting
    hard_accuracy = accuracy_score(y_test, hard_vote_predictions)
    hard_f1 = f1_score(y_test, hard_vote_predictions)
    
    # Print results
    print("\nEnsemble Model (Soft Voting) performance:")
    print(f"  Accuracy: {soft_accuracy:.4f}")
    print(f"  ROC AUC: {soft_roc_auc:.4f}")
    print(f"  F1 Score: {soft_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, soft_vote_predictions))
    
    print("\nEnsemble Model (Hard Voting) performance:")
    print(f"  Accuracy: {hard_accuracy:.4f}")
    print(f"  F1 Score: {hard_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, hard_vote_predictions))
    
    return {
        'Soft Voting': {
            'Probabilities': avg_probabilities,
            'Predictions': soft_vote_predictions,
            'Accuracy': soft_accuracy,
            'ROC AUC': soft_roc_auc,
            'F1 Score': soft_f1
        },
        'Hard Voting': {
            'Predictions': hard_vote_predictions,
            'Accuracy': hard_accuracy,
            'F1 Score': hard_f1
        }
    }

def visualize_model_comparison(results, ensemble_results, y_test):
    """Visualize model comparison"""
    # Create a DataFrame for model comparison
    model_names = list(results.keys()) + ['Ensemble (Soft)', 'Ensemble (Hard)']
    accuracies = [results[name]['Accuracy'] for name in results.keys()]
    accuracies.append(ensemble_results['Soft Voting']['Accuracy'])
    accuracies.append(ensemble_results['Hard Voting']['Accuracy'])
    
    f1_scores = [results[name]['F1 Score'] for name in results.keys()]
    f1_scores.append(ensemble_results['Soft Voting']['F1 Score'])
    f1_scores.append(ensemble_results['Hard Voting']['F1 Score'])
    
    roc_aucs = [results[name]['ROC AUC'] for name in results.keys()]
    roc_aucs.append(ensemble_results['Soft Voting']['ROC AUC'])
    roc_aucs.append(np.nan)  # Hard voting doesn't have ROC AUC
    
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'F1 Score': f1_scores,
        'ROC AUC': roc_aucs
    })
    
    # Save comparison to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    
    # Create a bar chart for model comparison
    plt.figure(figsize=(12, 8))
    
    # Plot metrics for each model
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.bar(x - width, accuracies, width, label='Accuracy', color='#3498db')
    plt.bar(x, f1_scores, width, label='F1 Score', color='#2ecc71')
    
    # Plot ROC AUC for models that have it
    roc_aucs_plot = []
    for auc in roc_aucs:
        if np.isnan(auc):
            roc_aucs_plot.append(0)
        else:
            roc_aucs_plot.append(auc)
    
    plt.bar(x + width, roc_aucs_plot, width, label='ROC AUC', color='#e74c3c')
    
    # Add labels and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i - width, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    for i, v in enumerate(roc_aucs):
        if not np.isnan(v):
            plt.text(i + width, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ensemble_comparison.png')
    plt.close()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Colors for the ROC curves
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot ROC curve for each model
    for i, name in enumerate(results.keys()):
        fpr, tpr, _ = roc_curve(y_test, results[name]['Probabilities'])
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{name} (AUC = {results[name]["ROC AUC"]:.3f})')
    
    # Plot ROC curve for ensemble (soft voting)
    fpr, tpr, _ = roc_curve(y_test, ensemble_results['Soft Voting']['Probabilities'])
    plt.plot(fpr, tpr, color='orange', lw=2,
             label=f'Ensemble (AUC = {ensemble_results["Soft Voting"]["ROC AUC"]:.3f})')
    
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
    plt.savefig('ensemble_roc_curves.png')
    plt.close()
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot confusion matrix for each model
    for i, name in enumerate(results.keys()):
        cm = confusion_matrix(y_test, results[name]['Predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Plot confusion matrix for ensemble (soft voting)
    cm = confusion_matrix(y_test, ensemble_results['Soft Voting']['Predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[3])
    axes[3].set_title('Confusion Matrix - Ensemble (Soft Voting)')
    axes[3].set_xlabel('Predicted')
    axes[3].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('ensemble_confusion_matrices.png')
    plt.close()

def save_ensemble_model(models):
    """Save the ensemble model"""
    # Save individual models
    for name, model in models.items():
        joblib.dump(model, f"final_{name.lower()}_model.pkl")
    
    # Save ensemble model info
    ensemble_info = {
        'model_names': list(models.keys()),
        'voting_method': 'soft'
    }
    
    joblib.dump(ensemble_info, "ensemble_model_info.pkl")
    
    print("\nEnsemble model saved successfully")

def main():
    # Load data
    X, y = load_data()
    if X is None or y is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Load best models
    models = load_best_models()
    
    # Train models
    trained_models = train_models(models, X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(trained_models, X_test, y_test)
    
    # Create ensemble predictions
    ensemble_results = create_ensemble_predictions(results, X_test, y_test)
    
    # Visualize model comparison
    visualize_model_comparison(results, ensemble_results, y_test)
    
    # Save ensemble model
    save_ensemble_model(trained_models)
    
    print("\nEnsemble model evaluation completed. Results saved to files.")

if __name__ == "__main__":
    main()
