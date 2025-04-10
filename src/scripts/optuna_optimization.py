import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Define objective functions for each model type

def objective_logistic_regression(trial, X, y):
    """Objective function for Logistic Regression optimization"""
    # Define hyperparameters to optimize
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    
    # Create model with suggested hyperparameters
    model = LogisticRegression(
        C=C, 
        solver=solver, 
        penalty=penalty, 
        class_weight=class_weight, 
        random_state=SEED,
        max_iter=1000
    )
    
    # Evaluate model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    return scores.mean()

def objective_random_forest(trial, X, y):
    """Objective function for Random Forest optimization"""
    # Define hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
    
    # Create model with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=SEED,
        n_jobs=-1
    )
    
    # Evaluate model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    return scores.mean()

def objective_gradient_boosting(trial, X, y):
    """Objective function for Gradient Boosting optimization"""
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    
    # Create model with suggested hyperparameters
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=SEED
    )
    
    # Evaluate model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    return scores.mean()

def objective_xgboost(trial, X, y):
    """Objective function for XGBoost optimization"""
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    gamma = trial.suggest_float('gamma', 0, 1)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 10)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 10)
    
    # Create model with suggested hyperparameters
    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Evaluate model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    return scores.mean()

def objective_lightgbm(trial, X, y):
    """Objective function for LightGBM optimization"""
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    num_leaves = trial.suggest_int('num_leaves', 20, 150)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 30)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    
    # Create model with suggested hyperparameters
    model = lgb.LGBMClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        class_weight='balanced',
        random_state=SEED,
        verbose=-1
    )
    
    # Evaluate model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    return scores.mean()

def objective_catboost(trial, X, y):
    """Objective function for CatBoost optimization"""
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    iterations = trial.suggest_int('iterations', 100, 500)
    depth = trial.suggest_int('depth', 4, 15)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)
    border_count = trial.suggest_int('border_count', 32, 255)
    bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1)
    
    # Create model with suggested hyperparameters
    model = CatBoostClassifier(
        learning_rate=learning_rate,
        iterations=iterations,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        bagging_temperature=bagging_temperature,
        random_seed=SEED,
        verbose=0
    )
    
    # Evaluate model using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    return scores.mean()

def optimize_models(X, y):
    """Run Optuna optimization for multiple models"""
    # Dictionary to store study results
    studies = {}
    best_scores = {}
    
    # Define models to optimize
    models = {
        'Logistic Regression': objective_logistic_regression,
        'Random Forest': objective_random_forest,
        'Gradient Boosting': objective_gradient_boosting,
        'XGBoost': objective_xgboost,
        'LightGBM': objective_lightgbm,
        'CatBoost': objective_catboost
    }
    
    # Run optimization for each model
    for model_name, objective_func in models.items():
        print(f"\nOptimizing {model_name}...")
        
        # Create a study for this model
        study = optuna.create_study(direction='maximize', study_name=model_name)
        
        # Run optimization
        study.optimize(
            lambda trial: objective_func(trial, X, y),
            n_trials=50,  # Adjust based on computational resources
            timeout=600,  # 10 minutes per model
            show_progress_bar=True
        )
        
        # Store study results
        studies[model_name] = study
        best_scores[model_name] = study.best_value
        
        print(f"Best {model_name} score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        # Save study
        joblib.dump(study, f"optuna_{model_name.replace(' ', '_').lower()}_study.pkl")
    
    return studies, best_scores

def create_best_models(studies, X_train, X_test, y_train, y_test):
    """Create and evaluate the best models based on Optuna studies"""
    # Dictionary to store trained models and their performance
    best_models = {}
    performance = {}
    
    # Train and evaluate each optimized model
    for model_name, study in studies.items():
        print(f"\nTraining optimized {model_name}...")
        
        # Create model with best parameters
        if model_name == 'Logistic Regression':
            model = LogisticRegression(
                **study.best_params,
                random_state=SEED,
                max_iter=1000
            )
        elif model_name == 'Random Forest':
            model = RandomForestClassifier(
                **study.best_params,
                random_state=SEED,
                n_jobs=-1
            )
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(
                **study.best_params,
                random_state=SEED
            )
        elif model_name == 'XGBoost':
            model = XGBClassifier(
                **study.best_params,
                random_state=SEED,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(
                **study.best_params,
                class_weight='balanced',
                random_state=SEED,
                verbose=-1
            )
        elif model_name == 'CatBoost':
            model = CatBoostClassifier(
                **study.best_params,
                random_seed=SEED,
                verbose=0
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        best_models[model_name] = model
        performance[model_name] = {
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'F1 Score': f1
        }
        
        # Print results
        print(f"{model_name} performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(model, f"best_{model_name.replace(' ', '_').lower()}_model.pkl")
    
    return best_models, performance

def visualize_optimization_results(studies):
    """Visualize Optuna optimization results"""
    for model_name, study in studies.items():
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image(f"optuna_{model_name.replace(' ', '_').lower()}_history.png")
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image(f"optuna_{model_name.replace(' ', '_').lower()}_param_importances.png")

def identify_best_models(performance):
    """Identify the best 3 models based on ROC AUC"""
    # Create DataFrame from performance dictionary
    performance_df = pd.DataFrame.from_dict(performance, orient='index')
    
    # Sort by ROC AUC
    sorted_df = performance_df.sort_values('ROC AUC', ascending=False)
    
    # Get top 3 models
    top_3_models = sorted_df.head(3)
    
    print("\nTop 3 Models Based on ROC AUC:")
    print(top_3_models)
    
    # Create bar chart of model performance
    plt.figure(figsize=(12, 8))
    
    # Plot ROC AUC for all models
    sns.barplot(x=sorted_df.index, y=sorted_df['ROC AUC'], palette='viridis')
    
    # Highlight top 3 models
    top_3_indices = sorted_df.head(3).index
    colors = ['gold', 'silver', 'brown']
    
    for i, model in enumerate(top_3_indices):
        model_idx = list(sorted_df.index).index(model)
        plt.bar(model_idx, sorted_df.loc[model, 'ROC AUC'], color=colors[i])
    
    plt.title('Model Performance Comparison (ROC AUC)', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('ROC AUC Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels for top 3 models
    for i, model in enumerate(top_3_indices):
        model_idx = list(sorted_df.index).index(model)
        plt.text(
            model_idx, 
            sorted_df.loc[model, 'ROC AUC'] + 0.01, 
            f"#{i+1}", 
            ha='center', 
            fontweight='bold', 
            color=colors[i]
        )
    
    plt.tight_layout()
    plt.savefig('top_models_comparison.png')
    plt.close()
    
    return top_3_models

def main():
    # Load data
    X, y = load_data()
    if X is None or y is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Run Optuna optimization
    studies, best_scores = optimize_models(X_train, y_train)
    
    # Create and evaluate best models
    best_models, performance = create_best_models(studies, X_train, X_test, y_train, y_test)
    
    # Visualize optimization results
    visualize_optimization_results(studies)
    
    # Identify best 3 models
    top_3_models = identify_best_models(performance)
    
    print("\nOptimization completed. Results saved to files.")

if __name__ == "__main__":
    main()
