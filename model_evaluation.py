import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def create_models():
    """Create a dictionary of models to evaluate"""
    models = {
        # 🔹 Logistic Regression: Strong regularization for better generalization
        'Logistic Regression': LogisticRegression(
            C=0.3, solver='liblinear', penalty='l1', class_weight='balanced', random_state=SEED
        ),

        # 🔹 K-Nearest Neighbors: Reducing overfitting with distance-based weighting
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=20, p=2
        ),

        # 🔹 Naive Bayes: Adjusted for better probability estimation
        'Naive Bayes': GaussianNB(var_smoothing=1e-8),

        # 🔹 Decision Tree: More depth and splitting to capture complex patterns
        'Decision Tree': DecisionTreeClassifier(
            criterion='entropy', max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=SEED
        ),

        # 🔹 Random Forest: More trees & depth for higher accuracy
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=3, min_samples_leaf=1,
            class_weight='balanced', random_state=SEED
        ),

        # 🔹 Support Vector Machine: Higher C, balanced class weights
        'Support Vector Machine': SVC(
            C=5.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=SEED
        ),

        # 🔹 Ridge Classifier: Optimized regularization for stability
        'Ridge Classifier': RidgeClassifier(alpha=0.3, class_weight='balanced'),

        # 🔹 Linear Discriminant Analysis: Optimized shrinkage
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),

        # 🔹 AdaBoost: More estimators & lower learning rate
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, learning_rate=0.05, random_state=SEED
        ),

        # 🔹 Gradient Boosting: Lower learning rate, more estimators
        'Gradient Boosting': GradientBoostingClassifier(
            learning_rate=0.03, n_estimators=250, max_depth=10, min_samples_split=3,
            min_samples_leaf=2, subsample=0.85, random_state=SEED
        ),

        # 🔹 Extra Trees Classifier: Higher estimators for robustness
        'Extra Trees Classifier': ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_split=4, random_state=SEED
        ),

        # 🔹 LightGBM: Optimized for best recall & precision
        'LightGBM': lgb.LGBMClassifier(
            colsample_bytree=0.9, learning_rate=0.03, max_depth=20, min_child_samples=5,
            n_estimators=250, num_leaves=90, subsample=0.85, class_weight='balanced',verbose=-1, random_state=SEED
        ),

        # 🔹 Multi-Layer Perceptron (MLP): More layers & epochs for better feature learning
        'Multi-Layer Perceptron (MLP)': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', alpha=0.0001,
            batch_size=16, learning_rate='adaptive', max_iter=500, random_state=SEED
        ),

        # 🔹 XGBoost: Tuned for high performance
        'XGBoost': XGBClassifier(
            colsample_bytree=0.8, learning_rate=0.03, max_depth=12, min_child_weight=4,
            n_estimators=250, subsample=0.85, gamma=0.1, reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=1, random_state=SEED
        ),

        # 🔹 CatBoost: Tuned for medical applications
        'CatBoost': CatBoostClassifier(
            iterations=250, learning_rate=0.03, depth=14, min_data_in_leaf=5,
            subsample=0.85, l2_leaf_reg=3, class_weights=[1, 4], random_state=SEED, verbose=0
        )
    }
    
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate models and store results in a DataFrame"""
    results = []
    trained_models = {}
    
    print("\nEvaluating models...")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Fit the model to the training data
        model.fit(X_train, y_train)
        
        # Store the trained model
        trained_models[name] = model
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Convert y_test and y_pred to integers before calculating accuracy
        accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
        
        # Get classification report for multi-class classification
        report = classification_report(y_test.astype(int), y_pred.astype(int), output_dict=True)
        
        # Collect results dynamically for each class
        model_result = {
            'Model': name,
            'Accuracy': accuracy,
        }
        
        # Add Precision, Recall, F1 score for each class dynamically
        for label in report.keys():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Exclude non-class keys
                model_result[f'Precision (Class {label})'] = report[label]['precision']
                model_result[f'Recall (Class {label})'] = report[label]['recall']
                model_result[f'F1 Score (Class {label})'] = report[label]['f1-score']
        
        # Add macro average metrics
        model_result['Macro Avg Precision'] = report['macro avg']['precision']
        model_result['Macro Avg Recall'] = report['macro avg']['recall']
        model_result['Macro Avg F1'] = report['macro avg']['f1-score']
        
        results.append(model_result)
    
    # Create a DataFrame from results
    results_df = pd.DataFrame(results)
    
    return results_df, trained_models

def visualize_results(results_df):
    """Visualize model performance"""
    # Create a bar chart of model accuracies
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values('Accuracy', ascending=False))
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.close()
    
    # Create a heatmap of model metrics
    metrics_cols = [col for col in results_df.columns if col != 'Model']
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        results_df.sort_values('Accuracy', ascending=False).set_index('Model')[metrics_cols],
        annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5
    )
    plt.title('Model Performance Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_performance_heatmap.png')
    plt.close()

def analyze_best_models(results_df, trained_models, X_test, y_test):
    """Analyze the best performing models"""
    # Get the top 3 models
    top_models = results_df.sort_values('Accuracy', ascending=False).head(3)
    print("\nTop 3 Models:")
    print(top_models[['Model', 'Accuracy', 'Macro Avg F1']])
    
    # Create confusion matrices for the top models
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (_, row) in enumerate(top_models.iterrows()):
        model_name = row['Model']
        model = trained_models[model_name]
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {model_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('top_models_confusion_matrices.png')
    plt.close()
    
    return top_models

def main():
    # Load data
    X, y = load_data()
    if X is None or y is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Create models
    models = create_models()
    
    # Evaluate models
    results_df, trained_models = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Sort results by accuracy
    sorted_results_df = results_df.sort_values(by='Accuracy', ascending=False)
    
    # Display the sorted results
    print("\nSorted Results by Accuracy:")
    print(sorted_results_df[['Model', 'Accuracy', 'Macro Avg F1']])
    
    # Save results to CSV
    sorted_results_df.to_csv('model_evaluation_results.csv', index=False)
    
    # Visualize results
    visualize_results(sorted_results_df)
    
    # Analyze best models
    top_models = analyze_best_models(sorted_results_df, trained_models, X_test, y_test)
    
    print("\nModel evaluation completed. Results saved to 'model_evaluation_results.csv'")
    print("Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
