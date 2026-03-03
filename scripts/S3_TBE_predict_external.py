import os
import joblib
import pandas as pd
import numpy as np
import ast 
from tqdm import tqdm

def load_all_domain_data(domain_dir):
    """
    Load all domain data (original features and labels)
    :param domain_dir: Domain data directory
    :return: Domain data dictionary {domain_name: (features, labels)}
    """
    domain_data = {}
    print(f"Loading domain data from: {domain_dir}")
    
    # Get all complete data files
    full_data_files = [f for f in os.listdir(domain_dir) if f.endswith('_full.csv')]
    print(f"Found {len(full_data_files)} complete data files")
    
    # Extract domain names from filenames
    domain_names = [f.replace('_full.csv', '') for f in full_data_files]
    
    for file_name, domain_name in zip(full_data_files, domain_names):
        try:
            # Build file path
            full_data_path = os.path.join(domain_dir, file_name)
            
            # Load complete data
            domain_df = pd.read_csv(full_data_path, index_col='Compound_CID')
            
            # Convert index to string
            domain_df.index = domain_df.index.astype(str)
            
            # Separate features and labels
            features = domain_df.drop('label', axis=1)
            labels = domain_df['label']
            
            domain_data[domain_name] = (features, labels)
            print(f"  Loaded {domain_name}: features shape={features.shape}, labels shape={labels.shape}")
        except Exception as e:
            print(f"  Failed to load {domain_name} ({file_name}): {str(e)}")
    
    return domain_data

def load_domain_models(model_dir):
    """
    Load all domain models
    :param model_dir: Model directory
    :return: Model dictionary {domain_name: model}
    """
    domain_models = {}
    print(f"Loading domain models from: {model_dir}")
    
    # Get all model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pkl")]
    print(f"Found {len(model_files)} model files")
    
    for model_file in tqdm(model_files, desc="Loading models"):
        try:
            # Parse domain name from filename
            # Filename format: model_{domain_name}_{model_type}.pkl
            # Remove prefix "model_" and suffix ".pkl"
            base_name = model_file.replace("model_", "").replace(".pkl", "")
            
            # Find last underscore position
            last_underscore = base_name.rfind('_')
            if last_underscore == -1:
                # No underscore, entire string as domain name
                domain_name = base_name
            else:
                # Extract domain name (remove model type)
                domain_name = base_name[:last_underscore]
            
            model_path = os.path.join(model_dir, model_file)
            model = joblib.load(model_path)
            
            domain_models[domain_name] = model
            print(f"  Loaded {domain_name} model: {type(model).__name__}")
        except Exception as e:
            print(f"  Failed to load model {model_file}: {str(e)}")
    
    return domain_models

def load_domain_features(feature_file):
    """
    Load feature lists for all domains from domain_feature_selection.csv
    :param feature_file: Feature selection result file path
    :return: Feature dictionary {domain_name: feature_names}
    """
    domain_features = {}
    print(f"Loading domain features from: {feature_file}")
    
    # Check if file exists
    if not os.path.exists(feature_file):
        print(f"  Error: File does not exist - {feature_file}")
        return domain_features
    
    # Load CSV file
    feature_df = pd.read_csv(feature_file, index_col=0)
    print(f"  Found feature selection results for {len(feature_df)} domains")
    
    # Process each domain
    for domain_name, row in feature_df.iterrows():
        try:
            # Safely convert string representation of list to actual Python list
            selected_features = ast.literal_eval(row['selected_features'])
            
            domain_features[domain_name] = selected_features
            print(f"  Loaded {domain_name}: {len(selected_features)} features")
        except Exception as e:
            print(f"  Failed to load features for {domain_name}: {str(e)}")
            print(f"  Raw data: {row['selected_features']}")
    
    return domain_features

def define_confidence_ad(model, X):
    """
    Confidence calculation based on prediction probability
    :param model: Trained model
    :param X: Input features
    :return: Confidence array
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

def predict_external_samples(domain_data, domain_models, domain_features):
    """
    Predict external samples for each domain and build summary table, also save normalized features
    :param domain_data: Domain data dictionary
    :param domain_models: Model dictionary
    :param domain_features: Feature dictionary
    :return: Summary DataFrame
    """
    # Get all sample IDs
    all_samples = set()
    for domain_name, (features, labels) in domain_data.items():
        sample_ids = features.index.astype(str).tolist()
        all_samples.update(features.index.tolist())
    all_samples = sorted(all_samples)
    
    # Create summary table
    result_df = pd.DataFrame(index=all_samples)
    
    # Add three columns for each domain
    domain_names = list(domain_data.keys())
    for domain in domain_names:
        result_df[f"{domain}_actual"] = np.nan
        result_df[f"{domain}_prediction"] = np.nan
        result_df[f"{domain}_confidence"] = np.nan
    
    # Fill actual labels
    for domain_name, (features, labels) in domain_data.items():
        common_samples = features.index.intersection(result_df.index)
        result_df.loc[common_samples, f"{domain_name}_actual"] = labels.loc[common_samples].values
    
    # Create directory for saving normalized features
    os.makedirs('TBEresults/external_AD', exist_ok=True)
    
    # Predict external samples for each domain
    print("\nStarting prediction of all samples...")
    for target_domain in tqdm(domain_names, desc="Processing domains"):
        print(f"\nProcessing target domain: {target_domain}")
        # Get target domain sample IDs
        target_domain_samples = set(domain_data[target_domain][0].index.astype(str))
        
        # Get target domain model and features
        model = domain_models.get(target_domain)
        if model is None:
            print(f"  Warning: Model not found for {target_domain}")
            continue
            
        feature_names = domain_features.get(target_domain)
        if feature_names is None:
            print(f"  Warning: Feature list not found for {target_domain}")
            continue
            
        # Collect samples from all other domains as external samples
        external_features_dict = {}  # Sample ID to features mapping

        for domain_name, (features, labels) in domain_data.items():
            # Skip target domain
            if domain_name == target_domain:
                continue

            # Ensure sample IDs are strings
            features.index = features.index.astype(str)

            # Ensure consistent feature order - maintain DataFrame format
            if isinstance(features, pd.DataFrame):
                # Create DataFrame with required features
                domain_features_df = features[feature_names].copy()

                # Ensure column names are feature names
                domain_features_df.columns = feature_names
            else:
                # If array, convert to DataFrame
                domain_features_df = pd.DataFrame(features, columns=feature_names, index=features.index)

            # Add samples (avoid duplicates, and exclude samples from target domain)
            for sample_id, row in domain_features_df.iterrows():
                # If sample not in target domain and not already added to external samples
                if sample_id not in target_domain_samples and sample_id not in external_features_dict:
                    external_features_dict[sample_id] = row

        if not external_features_dict:
            print(f"  Warning: No external samples found")
            continue

        # Build feature matrix from dictionary
        X_external = pd.DataFrame.from_dict(external_features_dict, orient='index')
        external_sample_ids = list(external_features_dict.keys())
        print(f"  External sample count: {len(external_sample_ids)}")

        # Predict
        try:
            # Get normalizer (if exists)
            preprocessor = None
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                print(f"  Found normalizer: {type(preprocessor).__name__}")
            
            # Apply normalization (if exists)
            if preprocessor is not None:
                print(f"  Applying normalization...")
                X_normalized = preprocessor.transform(X_external)
            else:
                print(f"  No normalizer, using raw features")
                X_normalized = X_external.values
            
            # Predict labels using model
            y_pred = model.predict(X_external)
            
            # Calculate confidence
            confidence = define_confidence_ad(model, X_external)
            
            # Create temporary DataFrame to save results
            temp_df = pd.DataFrame({
                'sample_id': external_sample_ids,
                f'{target_domain}_prediction': y_pred,
                f'{target_domain}_confidence': confidence
            }).set_index('sample_id')
            
            # Merge to summary table
            result_df.update(temp_df)
            
            # Save normalized features and confidence
            print("  Saving normalized features and confidence...")
            
            # Create metadata dictionary
            external_metadata = {
                'domain_name': target_domain,
                'features': X_normalized,  # Normalized feature matrix
                'feature_names': feature_names,  # Feature names
                'confidences': confidence,  # Confidence scores
                'sample_ids': external_sample_ids  # Sample IDs
            }
            
            # Save metadata
            metadata_path = f'TBEresults/external_AD/{target_domain}_external.pkl'
            joblib.dump(external_metadata, metadata_path)
            print(f"  External sample applicability domain data saved to: {metadata_path}")
            
            print(f"  Prediction complete: {len(y_pred)} external samples")
        except Exception as e:
            print(f"  Prediction failed: {str(e)}")
    
    return result_df

def predict_all_samples(domain_data, domain_models, domain_features):
    """
    Predict all samples (including samples from all domains) on each domain model
    :param domain_data: Domain data dictionary
    :param domain_models: Model dictionary
    :param domain_features: Feature dictionary
    :return: Summary DataFrame
    """
    # Get all sample IDs
    all_samples = set()
    for domain_name, (features, labels) in domain_data.items():
        all_samples.update(features.index.tolist())
    all_samples = sorted(all_samples)
    
    # Create summary table
    result_df = pd.DataFrame(index=all_samples)
    
    # Add three columns for each domain
    domain_names = list(domain_data.keys())
    for domain in domain_names:
        result_df[f"{domain}_actual"] = np.nan
        result_df[f"{domain}_prediction"] = np.nan
        result_df[f"{domain}_confidence"] = np.nan
    
    # Fill actual labels
    for domain_name, (features, labels) in domain_data.items():
        common_samples = features.index.intersection(result_df.index)
        result_df.loc[common_samples, f"{domain_name}_actual"] = labels.loc[common_samples].values
    
    # Create directory for saving all-sample applicability domain data
    os.makedirs('TBEresults/all_AD', exist_ok=True)
    
    # Predict all samples for each domain
    print("\nStarting prediction of all samples...")
    for target_domain in tqdm(domain_names, desc="Processing domains"):
        print(f"\nProcessing target domain: {target_domain}")
        
        # Get target domain model and features
        model = domain_models.get(target_domain)
        if model is None:
            print(f"  Warning: Model not found for {target_domain}")
            continue
            
        feature_names = domain_features.get(target_domain)
        if feature_names is None:
            print(f"  Warning: Feature list not found for {target_domain}")
            continue
            
        # Collect features for all samples
        all_features_dict = {}  # Sample ID to features mapping

        for domain_name, (features, labels) in domain_data.items():
            # Ensure sample IDs are strings
            features.index = features.index.astype(str)

            # Ensure consistent feature order - maintain DataFrame format
            if isinstance(features, pd.DataFrame):
                # Create DataFrame with required features
                domain_features_df = features[feature_names].copy()

                # Ensure column names are feature names
                domain_features_df.columns = feature_names
            else:
                # If array, convert to DataFrame
                domain_features_df = pd.DataFrame(features, columns=feature_names, index=features.index)

            # Add samples (avoid duplicates)
            for sample_id, row in domain_features_df.iterrows():
                if sample_id not in all_features_dict:
                    all_features_dict[sample_id] = row

        if not all_features_dict:
            print(f"  Warning: No samples found")
            continue

        # Build feature matrix from dictionary
        X_all = pd.DataFrame.from_dict(all_features_dict, orient='index')
        all_sample_ids = list(all_features_dict.keys())
        print(f"  Total sample count: {len(all_sample_ids)}")

        # Predict
        try:
            # Get normalizer (if exists)
            preprocessor = None
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                print(f"  Found normalizer: {type(preprocessor).__name__}")
            
            # Apply normalization (if exists)
            if preprocessor is not None:
                print(f"  Applying normalization...")
                X_normalized = preprocessor.transform(X_all)
            else:
                print(f"  No normalizer, using raw features")
                X_normalized = X_all.values
            
            # Predict labels using model
            y_pred = model.predict(X_all)
            
            # Calculate confidence
            confidence = define_confidence_ad(model, X_all)
            
            # Create temporary DataFrame to save results
            temp_df = pd.DataFrame({
                'sample_id': all_sample_ids,
                f'{target_domain}_prediction': y_pred,
                f'{target_domain}_confidence': confidence
            }).set_index('sample_id')
            
            # Merge to summary table
            result_df.update(temp_df)
            
            # Save all-sample applicability domain data
            print("  Saving all-sample applicability domain data...")
            
            # Create metadata dictionary
            full_metadata = {
                'domain_name': target_domain,
                'features': X_normalized,  # Normalized feature matrix
                'feature_names': feature_names,  # Feature names
                'confidences': confidence,  # Confidence scores
                'sample_ids': all_sample_ids  # Sample IDs
            }
            
            # Save metadata
            metadata_path = f'TBEresults/all_AD/{target_domain}_all.pkl'
            joblib.dump(full_metadata, metadata_path)
            print(f"  All-sample applicability domain data saved to: {metadata_path}")
            
            print(f"  Prediction complete: {len(y_pred)} samples")
        except Exception as e:
            print(f"  Prediction failed: {str(e)}")
    
    return result_df

def main():
    # Configure paths
    DATA_DIR = "TBEresults/domain_data"  # Domain data directory
    MODEL_DIR = "TBEresults"  # Model directory
    FEATURE_FILE = "TBEresults/domain_feature_selection.csv"  # Feature selection result file
    OUTPUT_DIR = "TBEresults"  # Output directory
    
    # Load all domain data
    domain_data = load_all_domain_data(DATA_DIR)
    
    # Load all domain models
    domain_models = load_domain_models(MODEL_DIR)
    
    # Load all domain feature lists - from CSV file
    domain_features = load_domain_features(FEATURE_FILE)
    
    # Predict external samples and build summary table
    print("\n===== Starting external sample prediction =====")
    external_prediction_results = predict_external_samples(domain_data, domain_models, domain_features)
    
    # Save external sample prediction summary table
    output_path = os.path.join(OUTPUT_DIR, "all_domain_predictions.csv")
    external_prediction_results.to_csv(output_path)
    print(f"  Saved external sample prediction summary table: {len(external_prediction_results)} rows x {len(external_prediction_results.columns)} columns")

    # Predict all samples and build summary table
    print("\n===== Starting all-sample prediction =====")
    all_prediction_results = predict_all_samples(domain_data, domain_models, domain_features)
    
    # Save all-sample prediction results
    alloutput_path = os.path.join(OUTPUT_DIR, "all_samples_predictions.csv")
    all_prediction_results.to_csv(alloutput_path)
    print(f"  Saved all-sample prediction summary table: {len(all_prediction_results)} rows x {len(all_prediction_results.columns)} columns")

    print("\nAll predictions complete!")

if __name__ == "__main__":
    main()