import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import StandardScaler  
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score, 
    balanced_accuracy_score, roc_curve, precision_recall_curve,
    average_precision_score  
)
from sklearn.base import clone
from tqdm import tqdm
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import parallel_backend
import matplotlib.ticker as ticker
import warnings
import logging
warnings.filterwarnings('ignore')

def configure_parallel_processing():
    """Configure parallel processing environment"""
    # Get SLURM allocated cores
    n_cores = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Configuring parallel processing: using {n_cores} cores")
    
    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
    
    return n_cores

def load_data(feature_file, domain_dir):
    """
    Load feature data and domain label data
    :param feature_file: Feature CSV file path
    :param domain_dir: Domain label file directory
    :return: Feature DataFrame, domain label dictionary {domain_name: (samples, labels)}
    """
    # Load feature data
    try:
        feature_df = pd.read_csv(feature_file)
        
        # Ensure CID column exists
        if 'Compound_CID' not in feature_df.columns:
            print("Error: Feature file missing 'Compound_CID' column")
            return None, {}
        
        # Process CID: convert to string and strip whitespace
        feature_df['Compound_CID'] = feature_df['Compound_CID'].astype(str).str.strip()
        feature_df.set_index('Compound_CID', inplace=True)
        print(f"Using Compound_CID as index (whitespace stripped)")
        
        # Skip first 6 columns (CID + 5 attribute columns)
        feature_data = feature_df.iloc[:, 5:]
        print(f"Feature data shape: {feature_data.shape}")
    except Exception as e:
        print(f"Failed to load feature file: {str(e)}")
        return None, {}
    
    # Load domain label data
    domain_data = {}
    
    # Ensure domain_dir exists
    if not os.path.exists(domain_dir):
        print(f"Error: Directory does not exist - {domain_dir}")
        return feature_data, {}
    
    # Get all Excel files
    excel_files = [f for f in os.listdir(domain_dir) if f.endswith('.xlsx')]
    print(f"Found {len(excel_files)} domain label files")
    
    for domain_file in excel_files:
        domain_name = os.path.splitext(domain_file)[0]
        file_path = os.path.join(domain_dir, domain_file)
        print(f"\nProcessing domain: {domain_name}")
        
        try:
            # Read Excel file
            domain_df = pd.read_excel(file_path, engine='openpyxl')
            
            # Process CID column
            if 'CID' not in domain_df.columns:
                print("  Error: Missing CID column")
                continue
            
            # Process CID: convert to string and strip whitespace
            domain_df['CID'] = domain_df['CID'].astype(str).str.strip()
            domain_df.set_index('CID', inplace=True)
            
            # Extract labels and convert to numeric
            if 'Final_Activity' not in domain_df.columns:
                print("  Error: Missing Final_Activity column")
                continue
            
            domain_df['label'] = domain_df['Final_Activity'].map({'active': 1, 'inactive': 0})
            
            # Ensure feature data and label data have same samples
            common_samples = feature_data.index.intersection(domain_df.index)
            
            # Report matching results
            print(f"  Matched samples: {len(common_samples)}")
            print(f"  Total labels: {len(domain_df)}")
            
            unmatched_samples = []
            
            # Save matching results
            if len(common_samples) > 0:
                domain_data[domain_name] = (
                    feature_data.loc[common_samples],
                    domain_df.loc[common_samples, 'label']
                )
            else:
                print("  Warning: No matching samples")
            
            # Check for mismatched samples
            if len(common_samples) < len(domain_df):
                unmatched_samples = domain_df.index.difference(feature_data.index)
                print(f"  Mismatched samples: {len(unmatched_samples)}")
            
            # Safely report mismatched samples
            if len(unmatched_samples) > 0:
                print(f"  Mismatched samples: {unmatched_samples[:10].tolist()}")
                
        except Exception as e:
            import traceback
            print(f"  Failed to process file: {str(e)}")
            traceback.print_exc()
    
    # Check after loading/preprocessing data
    print(f"Feature data loaded: shape={feature_data.shape}")
    print(f"Feature data type: {type(feature_data)}")
    print(f"Feature data column examples: {feature_data.columns[:5].tolist()}")

    print(f"\nSuccessfully loaded data for {len(domain_data)} domains")
    return feature_data, domain_data

def calculate_feature_count(n_samples):
    """
    Dynamically calculate feature selection count based on sample size
    :param n_samples: Number of training samples
    :return: Number of features
    """
    # Fixed 8 features for sample size < 100
    if n_samples < 100:
        return 8
    
    # Calculate feature count for samples ≥ 100
    n_features = 10 + (n_samples - 100) // 50
    
    # Limit maximum features to 20
    return min(n_features, 20)

def identify_categorical_features(feature_names):
    """
    Identify categorical features (based on fingerprint patterns in feature names)
    :param feature_names: List of all feature names
    :return: List of categorical feature indices
    """
    # Define fingerprint patterns
    fp_patterns = ["ExtFP", "EStateFP", "MACCSFP", "PubchemFP"]
    
    # Identify categorical features containing these patterns
    categorical_indices = []
    for i, name in enumerate(feature_names):
        if any(pattern in name for pattern in fp_patterns):
            categorical_indices.append(i)
    
    print(f"Identified {len(categorical_indices)} categorical features")
    return categorical_indices

def calculate_oversampling_target(y_train):
    class_counts = np.bincount(y_train)
    if len(class_counts) < 2:
        # Single class case, return dictionary with current count
        return {0: class_counts[0]}  # Assuming label is 0
    
    majority_label = np.argmax(class_counts)
    minority_label = np.argmin(class_counts)
    majority_count = class_counts[majority_label]
    minority_count = class_counts[minority_label]

    if minority_count == 0:
        # Minority class has 0 samples, return current class counts
        return {label: count for label, count in enumerate(class_counts)}
    
    # Calculate imbalance ratio (IR)
    imbalance_ratio = majority_count / minority_count
    print(f"  Imbalance Ratio (IR): {imbalance_ratio:.2f}")
    print(f"  Minority label: {minority_label}, sample count: {minority_count}")
    print(f"  Majority label: {majority_label}, sample count: {majority_count}")

    if imbalance_ratio <= 2:
        print("  IR ≤ 2: No oversampling")
        return {minority_label: minority_count}
    elif imbalance_ratio <= 5:
        target_minority = max(30, majority_count // 2)
        print(f"  2 < IR ≤ 5: Oversample minority to 1:2 (target: {target_minority})(original: {minority_count})")
    elif imbalance_ratio <= 10:
        target_minority = max(30, majority_count // 5)
        print(f"  5 < IR ≤ 10: Oversample minority to IR=5 (target: {target_minority})(original: {minority_count})")
    else:
        target_minority = max(30, majority_count // 8)
        print(f"  IR > 10: Oversample minority to IR=8 (target: {target_minority})(original: {minority_count})")
    
    if target_minority <= minority_count:
        print(f"  Warning: Calculated target ({target_minority}) not greater than original samples ({minority_count})")
        target_minority = minority_count + 10
        print(f"  Adjusted target to: {target_minority}")
    
    return {minority_label: target_minority}

def calculate_metrics(y_true, y_prob):
    """
    Calculate evaluation metrics (using default threshold 0.5)
    """
    # Predict using threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate positive sample ratio (as PR baseline)
    positive_rate = np.mean(y_true)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': precision_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'positive_rate': positive_rate,
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    }
    
    # Calculate curve data
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    metrics['roc_curve'] = (fpr, tpr)
    metrics['pr_curve'] = (precision, recall)
    
    return metrics

def calculate_feature_importance(X_train, y_train, n_features, domain_name):
    """
    Safely calculate feature importance (adaptive cross-validation based on sample size)
    :return: Feature importance DataFrame and selected feature list
    """
    feature_names = X_train.columns.tolist()
    n_samples = len(X_train)
    
    # Determine cross-validation folds based on sample size
    if n_samples < 100:
        n_folds = 3
        print(f"  {domain_name}: Small domain ({n_samples} samples), using 3-fold cross-validation for feature importance")
    elif n_samples < 300:
        n_folds = 5
        print(f"  {domain_name}: Medium domain ({n_samples} samples), using 5-fold cross-validation for feature importance")
    else:
        n_folds = 10
        print(f"  {domain_name}: Large domain ({n_samples} samples), using 10-fold cross-validation for feature importance")
    
    # Initialize importance accumulation arrays
    rf_importance_accum = np.zeros(len(feature_names))
    xgb_importance_accum = np.zeros(len(feature_names))
    
    # Create cross-validation object
    if n_folds > 1:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kf.split(X_train, y_train)
    else:
        # Single fold (full dataset)
        splits = [([np.arange(len(X_train))], [])]
    
    # Cross-validation loop
    for fold_idx, (train_idx, _) in enumerate(splits):
        print(f"    Processing fold {fold_idx+1}/{n_folds}...")
        
        # Get current fold training data
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        
        # Create base models (with default parameters)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42+fold_idx)
        xgb_model = XGBClassifier(
            n_estimators=100, 
            random_state=42+fold_idx,
            eval_metric='logloss'
        )
        
        # Train models
        rf_model.fit(X_fold, y_fold)
        xgb_model.fit(X_fold, y_fold)
        
        # Get feature importance
        rf_importance = rf_model.feature_importances_
        xgb_importance = xgb_model.feature_importances_
        
        # Normalize feature importance
        rf_importance_norm = rf_importance / rf_importance.sum()
        xgb_importance_norm = xgb_importance / xgb_importance.sum()
        
        # Accumulate importance
        rf_importance_accum += rf_importance_norm
        xgb_importance_accum += xgb_importance_norm
    
    # Calculate average importance
    rf_importance_avg = rf_importance_accum / n_folds
    xgb_importance_avg = xgb_importance_accum / n_folds
    
    # Calculate combined feature importance (average)
    combined_importance = (rf_importance_avg + xgb_importance_avg) / 2
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_importance_avg,
        'xgb_importance': xgb_importance_avg,
        'combined_importance': combined_importance
    })
    
    # Sort by combined importance
    importance_df.sort_values('combined_importance', ascending=False, inplace=True)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    # Select most important features
    selected_features = importance_df.head(n_features)['feature'].tolist()
    
    # Mark selected features
    importance_df['final_selected'] = importance_df['feature'].isin(selected_features)
    
    return importance_df, selected_features

def get_model_params(n_train_samples):
    """
    Dynamically set hyperparameters based on training sample size
    :param n_train_samples: Number of training samples
    :return: Model parameter dictionary
    """
    # Small domain (training samples < 100)
    if n_train_samples < 100:
        return {
            'XGBoost': {
                'n_estimators': [50, 100, 150],
                'max_depth': [2, 3],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9],
                'reg_lambda': [0.5, 1.0, 1.5]
            },
            'RandomForest': {
                'n_estimators': [50, 80, 100],
                'max_depth': [2, 3],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'max_features': [0.2, 0.3, 0.4]
            }
        }
    
    # Medium domain (100-300)
    elif n_train_samples < 300:
        return {
            'XGBoost': {
                'n_estimators': [100, 150, 200],
                'max_depth': [3, 4],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.7, 0.8],
                'reg_lambda': [0.5, 1.0, 1.5]
            },
            'RandomForest': {
                'n_estimators': [80, 100, 120],
                'max_depth': [3, 4],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'max_features': [0.3, 0.4, 0.5]
            }
        }
    
    # Large domain (>300)
    else:
        return {
            'XGBoost': {
                'n_estimators': [200, 300, 400],
                'max_depth': [4, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.5, 0.6],
                'reg_lambda': [0.5, 1.0, 1.5]
            },
            'RandomForest': {
                'n_estimators': [150, 200, 250],
                'max_depth': [4, 5],
                'min_samples_split': [10, 15],
                'min_samples_leaf': [2, 4],
                'max_features': [0.3, 0.4, 0.5]
            }
        }
    
# =========== Model Training Functions ====================
def train_domain_model(X_train, y_train, n_features, domain_name, n_cores):
    """
    Train domain model and select features (integrated XGB and RF feature importance)
    """
    # Step 1: Feature selection (calculate feature importance using both XGB and RF)
    print("  Performing feature selection (integrated XGB and RF cross-validation)...")
    importance_df, selected_features = calculate_feature_importance(
        X_train, y_train, n_features, domain_name
    )
    
    # Save feature importance results
    save_dir = "feature_importance/TBE"
    os.makedirs(save_dir, exist_ok=True)
    importance_path = os.path.join(save_dir, f'feature_importance_{domain_name}.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"  Feature importance saved to: {importance_path}")
    print(f"  Selected features: {selected_features[:5]}... total {len(selected_features)}")
    
    # Recalculate categorical feature indices on selected features
    X_train_selected = X_train[selected_features]
    selected_feature_names = X_train_selected.columns.tolist()
    categorical_indices = identify_categorical_features(selected_feature_names)
    print(f"  Identified {len(categorical_indices)} categorical features in selected features")

    # Calculate numerical feature indices
    all_indices = set(range(len(selected_features)))
    numerical_indices = list(all_indices - set(categorical_indices))

    # Create normalization processor
    if numerical_indices:
        # Use ColumnTransformer for numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_indices),
                ('cat', 'passthrough', categorical_indices)
            ],
            remainder='passthrough'
        )
    else:
        # Skip if only categorical features
        preprocessor = None
    
    # Get dynamic model parameters
    model_params = get_model_params(len(X_train))

    # Define models and parameter grids (using selected features)
    models = {
        'XGBoost': {
            'classifier': XGBClassifier(
                random_state=42,
                eval_metric='logloss'
            ),
            'params': {
                'classifier__n_jobs': [min(4, n_cores)],
                **{'classifier__' + k: v for k, v in model_params['XGBoost'].items()}
            }
        },
        'RandomForest': {
            'classifier': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_jobs': [min(4, n_cores)],
                **{'classifier__' + k: v for k, v in model_params['RandomForest'].items()}
            }
        }
    }
    
    best_model = None
    best_score = -1
    best_composite_score = -1
    best_model_name = ""
    best_cv_results = {}

    # Calculate oversampling strategy
    sampling_strategy = calculate_oversampling_target(y_train)
    
    for model_name, model_info in models.items():
        print(f"  Tuning {model_name} (feature subset)...")
        
        # Build Pipeline with oversampling
        if categorical_indices:
            print(f"  Using SMOTENC for mixed features")
            sampler = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy=sampling_strategy,
                random_state=42
            )
        else:
            print(f"  Using SMOTE for numerical features")
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42
        )
        
        # Modified Pipeline
        if preprocessor is not None:
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', sampler),
                ('classifier', model_info['classifier'])
            ])
        else:
            pipeline = ImbPipeline([
                ('smote', sampler),
                ('classifier', model_info['classifier'])
            ])
        
        try:
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=model_info['params'],
                scoring='f1_macro',
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                n_jobs=min(16, n_cores//2),
                verbose=1,
                return_train_score=True
            )
            
            with parallel_backend('loky', inner_max_num_threads=4):
                grid_search.fit(X_train_selected, y_train)

            model = grid_search.best_estimator_
            score = grid_search.best_score_
            
            # Get GridSearchCV results
            cv_f1_scores = []
            for i in range(grid_search.n_splits_):
                score_key = f'split{i}_test_score'
                fold_score = grid_search.cv_results_[score_key][grid_search.best_index_]
                cv_f1_scores.append(fold_score)
            print(f"    {model_name} GridSearchCV fold F1 scores average: {np.mean(cv_f1_scores):.4f}")
 
            # Print current model tuning results
            print(f"    {model_name} best hyperparameters: {grid_search.best_params_}")
            print(f"    {model_name} best model F1_score: {score:.4f}")
            
            # Recalculate cross-validation (multiple metrics)
            cv_results = evaluate_model_with_params(
                pipeline, grid_search.best_params_, X_train_selected, y_train, 
                n_splits=10, random_state=42
            )
            # Print recalculated fold F1 scores
            print(f"    {model_name} recalculated fold F1 scores average: {np.mean(cv_results['val_f1']):.4f}")
           
            # Calculate overfitting score
            overfitting_score = np.mean(cv_results['train_f1']) - np.mean(cv_results['val_f1'])
            print(f"    Overfitting score: {overfitting_score:.4f}")
            
            # Calculate composite score
            composite_score = score - overfitting_score * 0.5  # Balance F1 and overfitting

            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_score = score
                best_model = model
                best_model_name = model_name
                print(f"    => {model_name} becomes current best model (F1: {score:.4f}, composite: {composite_score:.4f})")
                # Save cross-validation results
                best_cv_results = cv_results
    
        except Exception as e:
            print(f"  Tuning {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if best_model is None:
        print("  Fallback: simple RandomForest")
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X_train_selected, y_train)
        best_model_name = "RandomForest (fallback)"
        logging.warning(f"Using fallback model for {domain_name}")
        # Set empty cv_results structure
        best_cv_results = {
            'train_f1': [],
            'train_auc_roc': [],
            'train_auc_pr': [],
            'train_accuracy': [],
            'train_balanced_accuracy': [],
            'train_sensitivity': [],
            'train_specificity': [],
            'train_precision': [],
            'train_mcc': [],
            'val_f1': [],
            'val_auc_roc': [],
            'val_auc_pr': [],
            'val_accuracy': [],
            'val_balanced_accuracy': [],
            'val_sensitivity': [],
            'val_specificity': [],
            'val_precision': [],
            'val_mcc': [],
        }
        # Set default composite score
        best_score = -1
        best_composite_score = -1
        # Mark as fallback model
        is_fallback_model = True
    else:
        is_fallback_model = False
    
    # Get final classifier
    if is_fallback_model:
        final_classifier = best_model
    else:
        final_classifier = best_model.named_steps['classifier']
    print(f"  Selected model: {best_model_name}, best score: {best_score:.4f}")
        
    # Get built-in feature importance
    if hasattr(final_classifier, 'feature_importances_'):
        final_importance = final_classifier.feature_importances_
    elif hasattr(final_classifier, 'coef_'):
        # For linear models
        if len(final_classifier.coef_.shape) > 1:
            final_importance = np.abs(final_classifier.coef_[0])
        else:
            final_importance = np.abs(final_classifier.coef_)
    else:
        final_importance = None
    
    if final_importance is not None:
        # Normalize
        final_importance_norm = final_importance / final_importance.sum()
        
        # Ensure importance array length matches selected features
        if len(final_importance_norm) != len(selected_features):
            print(f"  Warning: Best model feature count ({len(final_importance_norm)}) doesn't match selected features ({len(selected_features)})")
            min_len = min(len(final_importance_norm), len(selected_features))
            selected_features_subset = selected_features[:min_len]
            final_importance_norm = final_importance_norm[:min_len]
        else:
            selected_features_subset = selected_features
        
        # Add to importance_df
        importance_df['final_importance'] = 0.0
        for i, feature in enumerate(selected_features_subset):
            if feature in importance_df['feature'].values:
                idx = importance_df.index[importance_df['feature'] == feature][0]
                importance_df.at[idx, 'final_importance'] = final_importance_norm[i]

    # Add SHAP feature importance calculation
    if best_model is not None:
        try:
            # Extract trained model from Pipeline
            trained_classifier = best_model.named_steps['classifier']
            
            # Calculate SHAP values (efficient algorithm)
            explainer = shap.TreeExplainer(trained_classifier)
            
            # Calculate SHAP values for selected features
            sample_size = min(200, len(X_train_selected))
            sample_indices = np.random.choice(len(X_train_selected), sample_size, replace=False)
            X_train_sample = X_train_selected.iloc[sample_indices]
            shap_values = explainer.shap_values(X_train_sample)

            # Debug SHAP value structure
            print(f"  SHAP value type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"  List length: {len(shap_values)}")
                for i, sv in enumerate(shap_values):
                    print(f"    Element {i} shape: {sv.shape}")
            elif isinstance(shap_values, np.ndarray):
                print(f"  SHAP value shape: {shap_values.shape}")
            
            # Calculate SHAP importance
            if isinstance(shap_values, list):
                # List form: binary classification, take positive class SHAP values
                shap_abs = np.abs(shap_values[1])
                shap_importance = np.mean(shap_abs, axis=0)
            else:
                # Array form
                if len(shap_values.shape) == 3:
                    # Binary or multi-class
                    if shap_values.shape[2] == 2:
                        # Binary: take positive class
                        shap_abs = np.abs(shap_values[:, :, 1])
                        shap_importance = np.mean(shap_abs, axis=0)
                    else:
                        # Multi-class: average across classes
                        shap_abs = np.abs(shap_values).mean(axis=2)
                        shap_importance = np.mean(shap_abs, axis=0)
                else:
                    # Regression or binary classification alternative form
                    shap_abs = np.abs(shap_values)
                    shap_importance = np.mean(shap_abs, axis=0)
            
            # Normalize SHAP importance
            shap_importance_norm = shap_importance / shap_importance.sum()
            print(f"  Normalized SHAP importance shape: {shap_importance_norm.shape}")
            print(f"  Normalized SHAP importance: {shap_importance_norm}")

            # Ensure shap_importance_norm is 1D array
            if len(shap_importance_norm.shape) > 1:
                # Flatten if 2D
                shap_importance_norm = shap_importance_norm.flatten()
                print(f"  Flattened SHAP importance shape: {shap_importance_norm.shape}")
            
            # Add to feature importance DataFrame
            importance_df['shap_importance'] = 0.0
            
            # Create new DataFrame with only selected features
            selected_importance_df = importance_df[importance_df['feature'].isin(selected_features)].copy()
            
            print(f"  Selected features: {selected_features}")
            print(f"  SHAP importance array length: {len(shap_importance_norm)}")
            print(f"  Selected feature DataFrame rows: {len(selected_importance_df)}")
            
            # Ensure length matches
            if len(selected_features) != len(shap_importance_norm):
                print(f"  Warning: Selected feature count ({len(selected_features)}) doesn't match SHAP importance ({len(shap_importance_norm)})")
                min_len = min(len(selected_features), len(shap_importance_norm))
                selected_features = selected_features[:min_len]
                shap_importance_norm = shap_importance_norm[:min_len]
            
            # Assign SHAP importance to each selected feature
            for i, feature in enumerate(selected_features):
                if feature in selected_importance_df['feature'].values:
                    idx = selected_importance_df.index[selected_importance_df['feature'] == feature][0]
                    value = shap_importance_norm[i]
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            value = value.item()
                        else:
                            value = value[0]
                    selected_importance_df.at[idx, 'shap_importance'] = value
            
            # Merge SHAP importance back to original DataFrame
            importance_df['shap_importance'] = importance_df['shap_importance'].fillna(0.0)
            importance_df.update(selected_importance_df['shap_importance'])
            
            # Save to file
            importance_df.to_csv(importance_path, index=False)

            # Visualize shap importance
            visualize_shap_importance(X_train_sample, shap_values, selected_features, domain_name)
            
        except Exception as e:
            print(f"  SHAP analysis failed: {str(e)}")
            logging.warning(f"SHAP analysis failed for {domain_name}: {str(e)}")

    y_fulltrain_prob = best_model.predict_proba(X_train_selected)[:, 1]
    fulltrain_metrics = calculate_metrics(y_train, y_fulltrain_prob)
    
    # Get Pipeline processed data after training
    if best_model is not None and not is_fallback_model:
        if 'preprocessor' in best_model.named_steps:
            X_train_normalized = best_model.named_steps['preprocessor'].transform(X_train_selected)
            fitted_preprocessor = best_model.named_steps['preprocessor']
            # Print normalized data information
            print(f"  \nNormalized training set data example (first feature of 5 samples): {X_train_normalized[:5, 0]}")
            print(f"  Normalized training set feature mean: {np.mean(X_train_normalized, axis=0)[:5]}")
            print(f"  Normalized training set feature std: {np.std(X_train_normalized, axis=0)[:5]}")
        else:
            X_train_normalized = X_train_selected
            fitted_preprocessor = None
            print("  No normalization processor, using raw data")
    else:
        # Fallback model has no Pipeline
        X_train_normalized = X_train_selected
    
    # Save data for applicability domain calculation
    AD_DATA_DIR = 'TBEresults/ad_data'
    os.makedirs(AD_DATA_DIR, exist_ok=True)
    
    # Create training data structure with sample IDs
    train_data = {
        'features': X_train_normalized,
        'labels': y_train.values,
        'sample_ids': y_train.index.tolist()
    }
    
    # Save training set data
    joblib.dump(selected_features, f'{AD_DATA_DIR}/{domain_name}_feature_names.pkl')
    joblib.dump(train_data, f'{AD_DATA_DIR}/{domain_name}_train_data.pkl')
    print(f"  Training set data saved to: {AD_DATA_DIR}/{domain_name}_train_data.pkl (contains {len(train_data['sample_ids'])} sample IDs)")
    
    print("  Calculating training set confidence...")
    train_confidence = define_confidence_ad(best_model, X_train_selected)

    # Save data for applicability domain calculation
    ad_data = {
        'domain_name': domain_name,
        'categorical_indices': categorical_indices,
        'model_type': best_model_name,
        'preprocessor': fitted_preprocessor,
        'train_sample_ids': y_train.index.tolist()
    }
    ad_data['train_confidence'] = train_confidence
    print(f"  Training set confidence calculated (mean: {np.mean(train_confidence):.3f})")
    joblib.dump(ad_data, f'{AD_DATA_DIR}/{domain_name}_ad_data.pkl')
    print(f"  Applicability domain data saved to: TBEresults/ad_data/{domain_name}_ad_data.pkl")

    return best_model, best_model_name, selected_features, importance_df, best_cv_results, fulltrain_metrics, ad_data

def evaluate_model_with_params(pipeline, params, X, y, n_splits=10, random_state=42):
    """
    Recalculate cross-validation with given parameters
    """
    # Set model parameters
    model = clone(pipeline)
    model.set_params(**params)
    
    # Create cross-validation object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize result storage
    results = {
        'train_auc_roc': [], 'val_auc_roc': [],
        'train_auc_pr': [], 'val_auc_pr': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_balanced_accuracy': [], 'val_balanced_accuracy': [],
        'train_sensitivity': [], 'val_sensitivity': [],
        'train_specificity': [], 'val_specificity': [],
        'train_precision': [], 'val_precision': [],
        'train_f1': [], 'val_f1': [],
        'train_mcc': [], 'val_mcc': []
    }
    
    # Perform cross-validation
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Training set predictions
        y_train_pred = model.predict(X_train_fold)
        y_train_prob = model.predict_proba(X_train_fold)[:, 1]
        
        # Validation set predictions
        y_val_pred = model.predict(X_val_fold)
        y_val_prob = model.predict_proba(X_val_fold)[:, 1]
        
        # Calculate training set metrics
        results['train_auc_roc'].append(roc_auc_score(y_train_fold, y_train_prob))
        results['train_auc_pr'].append(average_precision_score(y_train_fold, y_train_prob))
        results['train_accuracy'].append(accuracy_score(y_train_fold, y_train_pred))
        results['train_balanced_accuracy'].append(balanced_accuracy_score(y_train_fold, y_train_pred))
        results['train_sensitivity'].append(recall_score(y_train_fold, y_train_pred))
        results['train_specificity'].append(calculate_specificity(y_train_fold, y_train_pred))
        results['train_precision'].append(precision_score(y_train_fold, y_train_pred))
        results['train_f1'].append(f1_score(y_train_fold, y_train_pred, average='macro'))
        results['train_mcc'].append(matthews_corrcoef(y_train_fold, y_train_pred))
        
        # Calculate validation set metrics
        results['val_auc_roc'].append(roc_auc_score(y_val_fold, y_val_prob))
        results['val_auc_pr'].append(average_precision_score(y_val_fold, y_val_prob))
        results['val_accuracy'].append(accuracy_score(y_val_fold, y_val_pred))
        results['val_balanced_accuracy'].append(balanced_accuracy_score(y_val_fold, y_val_pred))
        results['val_sensitivity'].append(recall_score(y_val_fold, y_val_pred))
        results['val_specificity'].append(calculate_specificity(y_val_fold, y_val_pred))
        results['val_precision'].append(precision_score(y_val_fold, y_val_pred))
        results['val_f1'].append(f1_score(y_val_fold, y_val_pred, average='macro'))
        results['val_mcc'].append(matthews_corrcoef(y_val_fold, y_val_pred))
    
    return results

def calculate_specificity(y_true, y_pred):
    """
    Calculate specificity
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def visualize_shap_importance(X_sample, shap_values, feature_names, domain_name):
    """
    Visualize SHAP feature importance (swarm plot)
    """
    # Create plot directory
    os.makedirs('feature_importance_plots/TBE', exist_ok=True)
    
    try:
        # Handle binary classification SHAP values (3D array)
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Use positive class SHAP values (index 1)
            shap_values_used = shap_values[:, :, 1]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            # Use positive class SHAP values
            shap_values_used = shap_values[1]
        else:
            shap_values_used = shap_values
        
        # SHAP swarm plot (show all features)
        plt.figure(figsize=(8, 10))
        # Create custom font size dictionary
        plot_params = {
            'font_size': 12,
            'axis_font_size': 14,
            'label_font_size': 14,
            'tick_font_size': 14,
            'color_bar_font_size': 14
        }
        
        # Use matplotlib rcParams to set global font size
        mpl.rcParams['font.size'] = plot_params['font_size']
        mpl.rcParams['axes.labelsize'] = plot_params['axis_font_size']
        mpl.rcParams['xtick.labelsize'] = plot_params['tick_font_size']
        mpl.rcParams['ytick.labelsize'] = plot_params['tick_font_size']

        shap.summary_plot(
            shap_values_used, 
            X_sample, 
            feature_names=feature_names,
            max_display=len(feature_names),  # Show all features
            plot_size=(8, 10),  # Adjust plot area size
            show=False
        )
        # Get current figure object for additional adjustments
        fig = plt.gcf()
        ax = plt.gca()
        
        # Modify Feature Value label font size
        for item in ax.get_children():
            # Find feature value label
            if hasattr(item, 'get_label') and item.get_label() == 'Feature value':
                item.set_size(plot_params['color_bar_font_size'])
        
        # Modify tick label size (High and Low)
        colorbar = fig.get_axes()[-1]  # Get right colorbar axis
        colorbar.tick_params(labelsize=plot_params['tick_font_size'])
        
        # Set colorbar title (Feature Value) font size
        colorbar.set_ylabel('Feature value', fontsize=plot_params['label_font_size'])
        
        plt.title(f'Domain-Specific TBE SHAP - {domain_name}', fontsize=16)
        plt.tight_layout()
        
        # Save to feature_importance_plots directory
        save_path = f'feature_importance_plots/TBE/TBEshap_{domain_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  SHAP swarm plot saved to: {save_path}")
        
        # Reset matplotlib parameters
        mpl.rcParams.update(mpl.rcParamsDefault)
    except Exception as e:
        print(f"  SHAP visualization failed: {str(e)}")
        # Ensure reset matplotlib parameters on error
        mpl.rcParams.update(mpl.rcParamsDefault)
        logging.warning(f"SHAP visualization failed for {domain_name}: {str(e)}")

def define_confidence_ad(model, X):
    """
    Confidence calculation based on prediction probability
    """
    # If model is Pipeline, get final classifier
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
    else:
        classifier = model

    if hasattr(classifier, 'predict_proba'):
        probas = model.predict_proba(X)
        # Take maximum probability as confidence
        confidence = np.max(probas, axis=1)
    else:
        # Fallback: use fixed value
        confidence = np.ones(X.shape[0])
    
    return confidence

def visualize_evaluation(test_metrics, domain_name):
    """
    visualize evaluation results
    """
    plot_data = {
        'domain': domain_name,
        'roc_data': None,
        'pr_data': None
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Domain-Specific TBE - {domain_name}', fontsize=16, fontweight='bold', y=0.98)
    
    roc_color = '#d2691e'  # darkorange
    prc_color = '#0066cc'  # blue
    diag_color = '#696969'  # dimgray
    
    # ROC
    if test_metrics.get('roc_curve'):
        fpr, tpr = test_metrics['roc_curve']
        roc_auc = test_metrics['auc_roc']
        
        # save ROC data
        plot_data['roc_data'] = {
            'fpr': fpr.tolist() if isinstance(fpr, np.ndarray) else fpr,
            'tpr': tpr.tolist() if isinstance(tpr, np.ndarray) else tpr,
            'auc': roc_auc,
            'thresholds': []
        }
        
        axes[0].plot(fpr, tpr, color=roc_color, 
                   lw=2, alpha=0.9, label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], color=diag_color, lw=1.5, linestyle='--')
        
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curve', fontsize=14)
        axes[0].legend(loc="lower right", fontsize=10)
        axes[0].grid(alpha=0.2, linestyle='--', color=diag_color)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    
    os.makedirs('evaluation_plots/TBE', exist_ok=True)
    plt.savefig(f'evaluation_plots/TBE/{domain_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    os.makedirs('evaluation_data/TBE', exist_ok=True)
    
    if plot_data['roc_data']:
        roc_df = pd.DataFrame({
            'fpr': plot_data['roc_data']['fpr'],
            'tpr': plot_data['roc_data']['tpr']
        })
        roc_df.to_csv(f'evaluation_data/TBE/{domain_name}_roc.csv', index=False)
    
    if plot_data['pr_data']:
        pr_df = pd.DataFrame({
            'recall': plot_data['pr_data']['recall'],
            'precision': plot_data['pr_data']['precision']
        })
        pr_df.to_csv(f'evaluation_data/TBE/{domain_name}_pr.csv', index=False)
    
    # save AUC value
    auc_data = {
        'domain': domain_name,
        'roc_auc': plot_data['roc_data']['auc'] if plot_data['roc_data'] else None,
        'pr_auc': plot_data['pr_data']['auc'] if plot_data['pr_data'] else None,
        'baseline': plot_data['pr_data']['baseline'] if plot_data['pr_data'] else None
    }

    auc_path = 'evaluation_data/TBE/auc_summary.csv'
    if not os.path.exists(auc_path):
        auc_df = pd.DataFrame(columns=['domain', 'roc_auc', 'pr_auc', 'baseline'])
    else:
        auc_df = pd.read_csv(auc_path)
    
    auc_df = auc_df[auc_df['domain'] != domain_name]
    new_row = pd.DataFrame([auc_data])
    auc_df = pd.concat([auc_df, new_row], ignore_index=True)
    auc_df.to_csv(auc_path, index=False)
    
    print(f"  evaluation plots save to: evaluation_plots/TBE/{domain_name}.png")
    print(f"  evaluation data(CSV) save to: evaluation_data/TBE/{domain_name}_*.csv")
    
    return plot_data  

def process_domains(domain_data, n_cores):
    """
    Process all domains
    """
    feature_selection_results = {}
    model_evaluation_results = {}
    
    for domain_name, (X, y) in tqdm(domain_data.items(), desc="Processing domains"):
        print(f"\nProcessing domain: {domain_name}")
        print(f"  Total samples: {len(X)}, Active: {sum(y==1)}, Inactive: {sum(y==0)}")
        
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"  Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Calculate feature selection count
        n_features = calculate_feature_count(len(X_train))
        print(f"  Selecting {n_features} features")
        
        # Train model and select features
        best_model, best_model_name, selected_features, importance_df, best_cv_results, fulltrain_metrics, ad_data = train_domain_model(X_train, y_train, n_features, domain_name, n_cores)
        print(f"  Selected features: {selected_features[:5]}... total {len(selected_features)}")
        
        # Calculate test set confidence
        X_test_selected = X_test[selected_features]                        
        test_confidence = define_confidence_ad(best_model, X_test_selected)

        preprocessor = ad_data.get('preprocessor')
        if preprocessor is not None:
            try:
                X_test_normalized = preprocessor.transform(X_test_selected)
            except Exception as e:
                print(f"  Test set normalization failed: {str(e)}")
                X_test_normalized = X_test_selected
        else:
            X_test_normalized = X_test_selected
            print("  No normalization processor, using raw data")

        print(f"    Test set confidence calculated (mean: {np.mean(test_confidence):.3f})")
        ad_data['test_confidence'] = test_confidence
        ad_data['test_sample_ids'] = y_test.index.tolist()
        
        # Save test set data
        AD_DATA_DIR = 'TBEresults/ad_data'
        joblib.dump(ad_data, f'{AD_DATA_DIR}/{domain_name}_ad_data.pkl')
        # Create test set data structure with sample IDs
        test_data = {
            'features': X_test_normalized,
            'labels': y_test.values,
            'sample_ids': y_test.index.tolist()
        }
        # Save test set data
        joblib.dump(test_data, f'{AD_DATA_DIR}/{domain_name}_test_data.pkl')
        print(f"  Test set data saved to: {AD_DATA_DIR}/{domain_name}_test_data.pkl (contains {len(test_data['sample_ids'])} sample IDs)")

        # Test set evaluation
        y_prob = best_model.predict_proba(X_test_selected)[:, 1]
        test_metrics = calculate_metrics(y_test, y_prob)
        print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"  Test AUC-PR: {test_metrics['auc_pr']:.4f}")
        
        # Save results
        feature_selection_results[domain_name] = {
            'selected_features': selected_features,
            'train_samples': X_train.index.tolist(),
            'test_samples': X_test.index.tolist()
        }
        
        model_evaluation_results[domain_name] = {
            'cv_results': best_cv_results,
            'test_metrics': test_metrics,
            'fulltrain_metrics': fulltrain_metrics,
            'model': best_model,
            'best_model_name': best_model_name,
            'ad_data': ad_data
        }

        # Visualization
        visualize_evaluation(test_metrics, domain_name)
    
    return feature_selection_results, model_evaluation_results

def integrate_features(feature_selection_results):
    """
    Integrate features selected by all domains
    """
    all_features = set()
    for domain, results in feature_selection_results.items():
        all_features.update(results['selected_features'])
    
    print(f"\nIntegrated features: selected {len(all_features)} features")
    return list(all_features)

def save_all_results(feature_selection_results, model_evaluation_results, integrated_features, domain_data):
    """
    Save all results
    """
    # Create main results directory
    os.makedirs('TBEresults', exist_ok=True)
    
    # Save feature selection results
    feature_df = pd.DataFrame.from_dict(feature_selection_results, orient='index')
    feature_df.to_csv('TBEresults/domain_feature_selection.csv')
    
    # Save model evaluation results
    metrics_data = []
    for domain, results in model_evaluation_results.items():
        cv_results = results['cv_results']
        test_metrics = results['test_metrics']
        fulltrain_metrics = results['fulltrain_metrics']
        
        # Basic metrics
        domain_metrics = {
            'domain': domain,
            'cv_train_f1_mean': np.mean(cv_results['train_f1']),
            'cv_train_auc_roc_mean': np.mean(cv_results['train_auc_roc']),
            'cv_train_auc_pr_mean': np.mean(cv_results['train_auc_pr']),
            'cv_train_accuracy_mean': np.mean(cv_results['train_accuracy']),
            'cv_train_balanced_accuracy_mean': np.mean(cv_results['train_balanced_accuracy']),
            'cv_train_sensitivity_mean': np.mean(cv_results['train_sensitivity']),
            'cv_train_specificity_mean': np.mean(cv_results['train_specificity']),
            'cv_train_precision_mean': np.mean(cv_results['train_precision']),
            'cv_train_mcc_mean': np.mean(cv_results['train_mcc']),
            'cv_train_f1_std': np.std(cv_results['train_f1']),
            'cv_train_auc_roc_std': np.std(cv_results['train_auc_roc']),
            'cv_train_auc_pr_std': np.std(cv_results['train_auc_pr']),
            'cv_train_accuracy_std': np.std(cv_results['train_accuracy']),
            'cv_train_balanced_accuracy_std': np.std(cv_results['train_balanced_accuracy']),
            'cv_train_sensitivity_std': np.std(cv_results['train_sensitivity']),
            'cv_train_specificity_std': np.std(cv_results['train_specificity']),
            'cv_train_precision_std': np.std(cv_results['train_precision']),
            'cv_train_mcc_std': np.std(cv_results['train_mcc']),
            'cv_val_f1_mean': np.mean(cv_results['val_f1']),
            'cv_val_auc_roc_mean': np.mean(cv_results['val_auc_roc']),
            'cv_val_auc_pr_mean': np.mean(cv_results['val_auc_pr']),
            'cv_val_accuracy_mean': np.mean(cv_results['val_accuracy']),
            'cv_val_balanced_accuracy_mean': np.mean(cv_results['val_balanced_accuracy']),
            'cv_val_sensitivity_mean': np.mean(cv_results['val_sensitivity']),
            'cv_val_specificity_mean': np.mean(cv_results['val_specificity']),
            'cv_val_precision_mean': np.mean(cv_results['val_precision']),
            'cv_val_mcc_mean': np.mean(cv_results['val_mcc']),
            'cv_val_f1_std': np.std(cv_results['val_f1']),
            'cv_val_auc_roc_std': np.std(cv_results['val_auc_roc']),
            'cv_val_auc_pr_std': np.std(cv_results['val_auc_pr']),
            'cv_val_accuracy_std': np.std(cv_results['val_accuracy']),
            'cv_val_balanced_accuracy_std': np.std(cv_results['val_balanced_accuracy']),
            'cv_val_sensitivity_std': np.std(cv_results['val_sensitivity']),
            'cv_val_specificity_std': np.std(cv_results['val_specificity']),
            'cv_val_precision_std': np.std(cv_results['val_precision']),
            'cv_val_mcc_std': np.std(cv_results['val_mcc']),
            'fulltrain_f1_score': fulltrain_metrics['f1_score'],
            'fulltrain_auc_roc': fulltrain_metrics['auc_roc'],
            'fulltrain_auc_pr': fulltrain_metrics['auc_pr'],
            'fulltrain_accuracy': fulltrain_metrics['accuracy'],
            'fulltrain_balanced_accuracy': fulltrain_metrics['balanced_accuracy'],
            'fulltrain_sensitivity': fulltrain_metrics['sensitivity'],
            'fulltrain_specificity': fulltrain_metrics['specificity'],
            'fulltrain_precision': fulltrain_metrics['precision'],
            'fulltrain_mcc': fulltrain_metrics['mcc'],
            'test_f1_score': test_metrics['f1_score'],
            'test_auc_roc': test_metrics['auc_roc'],
            'test_auc_pr': test_metrics['auc_pr'],
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_sensitivity': test_metrics['sensitivity'],
            'test_specificity': test_metrics['specificity'],
            'test_precision': test_metrics['precision'],
            'test_mcc': test_metrics['mcc']
        }
        
        metrics_data.append(domain_metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv('TBEresults/domain_model_metrics.csv', index=False)
    
    # Save integrated features
    pd.Series(integrated_features).to_csv('TBEresults/integrated_features.csv', index=False)
    
    # Save models
    for domain, results in model_evaluation_results.items():
        joblib.dump(results['model'], f"TBEresults/model_{domain}_{results['best_model_name']}.pkl")

    # Create domain data directory
    os.makedirs('TBEresults/domain_data', exist_ok=True)
    
    # Save each domain's original data and split information
    domain_info = {}
    
    for domain_name, (X, y) in domain_data.items():
        # Save original data
        domain_df = pd.concat([X, y], axis=1)
        domain_df.index.name = 'Compound_CID'
        domain_df.to_csv(f'TBEresults/domain_data/{domain_name}_full.csv', index=True)
        
        # Save split information
        if domain_name in feature_selection_results:
            results = feature_selection_results[domain_name]
            split_df = pd.DataFrame({
                'sample_id': results['train_samples'] + results['test_samples'],
                'split': ['train'] * len(results['train_samples']) + ['test'] * len(results['test_samples'])
            })
            split_df.to_csv(f'TBEresults/domain_data/{domain_name}_split.csv', index=False)
        
        # Record domain information
        domain_info[domain_name] = {
            'samples': len(X),
            'active': sum(y == 1),
            'inactive': sum(y == 0),
            'features': len(integrated_features)
        }
    
    # Save domain information
    pd.DataFrame.from_dict(domain_info, orient='index').to_csv('TBEresults/domain_data/domain_info.csv')
    
    print("All results saved to TBEresults directory")

def main():
    n_cores = configure_parallel_processing()
    logging.info(f"Starting domain modeling with {n_cores} cores")
    
    # File path configuration
    feature_file = 'path/to/allDescriptors_filled.csv'
    domain_dir = 'path/to/domain_dir'
    
    # Load data
    print("Loading data...")
    feature_data, domain_data = load_data(feature_file, domain_dir)
    print(f"Loaded data for {len(domain_data)} domains")
    
    # Process all domains
    print("\nStarting domain processing...")
    feature_selection_results, model_evaluation_results = process_domains(domain_data, n_cores)
    
    # Integrate features
    print("\nIntegrating features...")
    integrated_features = integrate_features(feature_selection_results)
    
    # Save results
    print("\nSaving results...")
    save_all_results(feature_selection_results, model_evaluation_results, integrated_features, domain_data)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()