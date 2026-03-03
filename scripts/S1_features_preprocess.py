import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import hashlib
from scipy.stats import mode
import os

def read_data(file_path):
    """Read data file and handle missing value representations"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.rds'):
        # Requires pyreadr package: pip install pyreadr
        import pyreadr
        df = pyreadr.read_r(file_path)[None]
    else:
        raise ValueError("Unsupported file format. Please use CSV or RDS files")
    
    # Unify missing value representations
    df.replace(['NA', 'Inf', '-Inf', 'inf', '-inf', 'NaN', 'nan', np.inf, -np.inf], 
               np.nan, inplace=True)
    
    return df

def remove_constant_features(feature_df, threshold=0.8):
    """Remove features with constant or near-constant values"""
    result = []
    for col in feature_df.columns:
        non_na_col = feature_df[col].dropna()
        if len(non_na_col) == 0:
            result.append(False)  # Remove columns with all NA
        else:
            max_freq = non_na_col.value_counts(normalize=True).max()
            result.append(max_freq < threshold)
    
    return feature_df.loc[:, result]

def remove_low_absolute_variance_features(feature_df, variance_threshold=0.001):
    """Remove features with low absolute variance"""
    feature_df = pd.DataFrame(feature_df)
    var_values = []
    
    for i, col in enumerate(feature_df.columns):
        col_data = feature_df.iloc[:, i]
        non_na_col = col_data.dropna()
        
        # Handle special values and invalid cases
        non_na_col = non_na_col[np.isfinite(non_na_col)]
        n_valid = len(non_na_col)
        
        # Safe variance calculation
        if n_valid < 2:
            var_values.append(0)
        else:
            col_variance = non_na_col.var()
            if np.isnan(col_variance) or not np.isfinite(col_variance):
                var_values.append(0)
            else:
                var_values.append(col_variance)
    
    # Apply absolute variance threshold
    keep_cols = [var_val >= variance_threshold and not np.isnan(var_val) 
                 for var_val in var_values]
    
    return feature_df.loc[:, keep_cols]

def remove_correlated_features(feature_df, cor_threshold=0.95):
    """Remove highly correlated features"""
    if feature_df.shape[1] < 2:
        return feature_df
    
    # 1. Calculate missing ratio and variance for each feature
    missing_ratio = feature_df.isna().mean()
    feature_var = feature_df.apply(lambda x: x.dropna().var() if len(x.dropna()) >= 2 else 0)
    
    # 2. Calculate correlation matrix
    cor_matrix = feature_df.corr(method='pearson', min_periods=1)
    
    # 3. Get highly correlated feature pairs
    high_cor_pairs = []
    for i in range(len(cor_matrix.columns)):
        for j in range(i+1, len(cor_matrix.columns)):
            cor_value = cor_matrix.iloc[i, j]
            if not np.isnan(cor_value) and abs(cor_value) > cor_threshold:
                high_cor_pairs.append((i, j, cor_matrix.columns[i], cor_matrix.columns[j]))
    
    if len(high_cor_pairs) == 0:
        return feature_df
    
    # 4. Create correlation groups
    cor_groups = []
    
    for i, j, col1, col2 in high_cor_pairs:
        found_group = False
        for group in cor_groups:
            if col1 in group or col2 in group:
                group.update([col1, col2])
                found_group = True
                break
        
        if not found_group:
            cor_groups.append(set([col1, col2]))
    
    # 5. For each correlation group, select features to keep
    to_keep = []
    to_remove = []
    
    for group in cor_groups:
        if len(group) > 1:
            group = list(group)
            # Calculate metrics for each feature in the group
            group_missing = missing_ratio[group]
            group_var = feature_var[group]
            
            # Find features with minimum missing ratio
            min_missing = group_missing.min()
            candidates = group_missing[group_missing == min_missing].index.tolist()
            
            # If multiple features have the same minimum missing ratio, 
            # select the one with maximum variance
            if len(candidates) > 1:
                candidate_var = group_var[candidates]
                best_candidate = candidate_var.idxmax()
            else:
                best_candidate = candidates[0]
            
            # Add to keep list
            to_keep.append(best_candidate)
            
            # Mark other features in the group for removal
            to_remove_group = [col for col in group if col != best_candidate]
            to_remove.extend(to_remove_group)
    
    # 6. Remove selected features
    if len(to_remove) > 0:
        print(f"In {len(cor_groups)} correlation groups, removed {len(to_remove)} highly correlated features")
        print(f"Retained features: {', '.join(to_keep[:10])}")
        if len(to_keep) > 10:
            print(f"  Showing first 10 features, total {len(to_keep)} retained")
        return feature_df.drop(columns=to_remove)
    
    return feature_df

def detect_duplicate_samples(df, property_cols_count=6):
    """Detect samples with identical feature columns"""
    print("Checking for samples with identical feature columns...")
    
    # Extract feature columns (columns after the first property_cols_count columns)
    feature_data = df.iloc[:, property_cols_count:]
    
    # Calculate hash for each row
    def row_hash(row):
        return hashlib.md5(str(row.values).encode()).hexdigest()
    
    feature_hashes = feature_data.apply(row_hash, axis=1)
    
    # Add hash column to original dataframe
    df_temp = df.copy()
    df_temp['FeatureHash'] = feature_hashes
    
    # Calculate count for each feature combination
    hash_counts = df_temp['FeatureHash'].value_counts()
    
    # Identify duplicate samples
    duplicate_hashes = hash_counts[hash_counts > 1].index
    duplicate_count = len(df_temp[df_temp['FeatureHash'].isin(duplicate_hashes)])
    
    if duplicate_count > 0:
        print(f"Warning: Found {duplicate_count} samples with identical feature columns")
        
        # Identify duplicate sample groups
        duplicate_groups = []
        for hash_val in duplicate_hashes:
            group_samples = df_temp[df_temp['FeatureHash'] == hash_val]
            cids = group_samples.iloc[:, 0].unique()  # Assuming CID is in first column
            duplicate_groups.append({
                'Sample_Count': len(group_samples),
                'CIDs': ', '.join(map(str, cids[:5]))
            })
        
        # Show sample duplicate group information
        print("Sample duplicate group information:")
        for i, group in enumerate(duplicate_groups[:5]):
            print(f"  Group {i+1}: {group['Sample_Count']} samples, CIDs: {group['CIDs']}")
        
        if len(duplicate_groups) > 5:
            print(f"  Showing first 5 groups, total {len(duplicate_groups)} groups")
        
        # Create duplicate sample report
        duplicate_report = df_temp[df_temp['FeatureHash'].isin(duplicate_hashes)].copy()
        duplicate_report = duplicate_report.groupby('FeatureHash').agg({
            df_temp.columns[0]: lambda x: ', '.join(map(str, x)),  # CID column
            'FeatureHash': 'count'
        }).rename(columns={df_temp.columns[0]: 'CIDs', 'FeatureHash': 'Count'})
        
        return duplicate_report
    else:
        print("✓ All samples have unique feature columns")
        return None

def knn_imputation_with_pca(df, feature_columns, k=20, n_components=100):
    """Perform KNN imputation with PCA for all missing values"""
    # Separate property columns and feature columns
    id_columns = df.columns[:6]
    properties = df[id_columns].copy()
    X = df[feature_columns].copy()
    
    # =========== Added: Identify binary classification features (0/1) ===========
    print("\n=== Identifying binary classification features ===")
    
    # Identify binary classification features (0/1)
    binary_features = []
    for col in feature_columns:
        unique_vals = X[col].dropna().unique()
        # Detect binary features: values contain only 0 and 1 (or include NaN)
        if set(unique_vals).issubset({0, 1, np.nan}) and (0 in unique_vals or 1 in unique_vals):
            binary_features.append(col)
    
    # Output identification results
    print(f"Identified {len(binary_features)} binary classification features (0/1):")
    if binary_features:
        print(", ".join(binary_features[:min(5, len(binary_features))]))
        if len(binary_features) > 5:
            print(f"  Showing first 5, total {len(binary_features)} features")
    else:
        print("No binary features identified")
    
    # Precompute global default value for each feature
    global_defaults = {}
    for col in feature_columns:
        if col in binary_features:
            # Binary features: use mode
            mode_vals = X[col].mode()
            global_defaults[col] = mode_vals.iloc[0] if not mode_vals.empty else 0
        else:
            # Continuous features: use median
            global_defaults[col] = X[col].median()
    # =========== End of binary feature identification ===========
    
    # 1. Create temporary imputed version for PCA
    temp_imputer = SimpleImputer(strategy='median')
    X_temp = pd.DataFrame(temp_imputer.fit_transform(X), 
                          columns=X.columns, 
                          index=X.index)
    
    # 2. Data standardization (important for distance calculation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_temp)
    
    # 3. PCA dimensionality reduction (improves distance calculation efficiency and robustness)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Calculate K nearest neighbors for all samples
    knn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1)  # +1 to include self
    knn.fit(X_pca)
    
    # Get neighbor indices for each sample (excluding self)
    _, neighbor_indices = knn.kneighbors(X_pca)
    neighbor_indices = neighbor_indices[:, 1:]  # Exclude first neighbor (self)
    
    # 5. Impute missing values (modified to handle binary features)
    X_filled = X.copy()
    
    # Record imputation statistics for each sample
    sample_stats = []
    
    for i in range(len(X)):
        current_sample = X.iloc[i]
        
        # Calculate missing features for current sample
        missing_mask = current_sample.isna()
        missing_features = missing_mask[missing_mask].index.tolist()
        n_missing = len(missing_features)
        
        if n_missing == 0:
            # No missing values, skip
            continue
            
        # Get neighbor indices for current sample
        neighbors = neighbor_indices[i]
        
        for feature in missing_features:
            # Extract feature values from neighbors (use only non-missing values)
            neighbor_values = X.iloc[neighbors][feature].dropna().values
            
            if len(neighbor_values) > 0:
                if feature in binary_features:
                    # === Binary feature handling: use mode of neighbors ===
                    # Calculate most frequent value in neighbors
                    values, counts = np.unique(neighbor_values, return_counts=True)
                    most_common_value = values[np.argmax(counts)]
                    
                    # Ensure imputed value is integer 0 or 1
                    fill_value = int(most_common_value)
                else:
                    # Continuous features: use mean of neighbors
                    fill_value = np.mean(neighbor_values)
            else:
                # Use global default when all neighbors are missing
                fill_value = global_defaults[feature]
                
            X_filled.loc[i, feature] = fill_value
        
        sample_stats.append({
            'Sample': properties.iloc[i, 0],  # Sample name in first column
            'MissingFeatures': n_missing,
            'MissingRatio': n_missing / len(feature_columns)
        })
        
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{len(X)} samples: {n_missing} missing features")
    
    # 6. Merge properties and imputed features
    result_df = pd.concat([properties, X_filled], axis=1)
    
    # 7. Generate imputation statistics
    stats_df = pd.DataFrame(sample_stats)
    
    return result_df, stats_df

def main():
    # File paths
    input_file = "path/to/allDescriptors.csv"
    output_processed = "path/to/allDescriptors_processed.csv"
    output_filled = "path/to/allDescriptors_filled.csv"
    output_stats = "path/to/imputation_stats.csv"
    duplicate_report_path = "path/to/duplicate_samples_report.csv"
    
    # Set parameters
    CONSTANT_THRESH = 0.8
    VARIANCE_THRESH = 0.001
    COR_THRESH = 0.95
    KNN_K = 20
    PCA_COMPONENTS = 100
    
    # 1. Read data
    print("Reading data file...")
    df = read_data(input_file)
    original_shape = df.shape
    print(f"Original data shape: {original_shape[0]} rows, {original_shape[1]} columns")
    
    # 2. Separate property columns (first 6) and feature columns
    properties = df.iloc[:, :6]
    features = df.iloc[:, 6:]
    print(f"Properties: {properties.shape[1]} columns, Features: {features.shape[1]} columns")
    
    # 3. Feature filtering
    print("\n" + "="*50)
    print("FEATURE FILTERING")
    print("="*50)
    
    # Step 1: Remove constant features
    print("\n1. Removing constant features...")
    filtered_step1 = remove_constant_features(features, CONSTANT_THRESH)
    print(f"   After removing constant features: {filtered_step1.shape[1]} features")
    
    # Step 2: Remove low variance features
    print("\n2. Removing low variance features...")
    filtered_step2 = remove_low_absolute_variance_features(filtered_step1, VARIANCE_THRESH)
    print(f"   After removing low variance features: {filtered_step2.shape[1]} features")
    
    # Step 3: Remove highly correlated features
    print("\n3. Removing highly correlated features...")
    filtered_step3 = remove_correlated_features(filtered_step2, COR_THRESH)
    print(f"   After removing highly correlated features: {filtered_step3.shape[1]} features")
    
    # 4. Reconstruct complete dataset
    processed_data = pd.concat([properties, filtered_step3], axis=1)
    print(f"\nProcessed data shape: {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")
    print(f"Retained {filtered_step3.shape[1]} features after filtering")
    
    # 5. Check for duplicate samples
    duplicate_report = detect_duplicate_samples(processed_data, property_cols_count=6)
    if duplicate_report is not None:
        duplicate_report.to_csv(duplicate_report_path, index=False)
        print(f"Duplicate sample report saved to: {duplicate_report_path}")
        
        # Optional: Ask for user confirmation to continue
        # Uncomment the following lines to enable user interaction
        # response = input("Duplicate samples found. Continue processing? (y/n): ")
        # if response.lower() != 'y':
        #     print("Processing terminated by user")
        #     return
    
    # 6. Save processed data
    print(f"\nSaving processed data to: {output_processed}")
    processed_data.to_csv(output_processed, index=False)
    
    # 7. Perform KNN imputation
    print("\n" + "="*50)
    print("K-NEAREST NEIGHBORS IMPUTATION")
    print("="*50)
    
    # Determine feature columns for imputation
    imputation_features = processed_data.columns[6:]
    
    print(f"Starting imputation with k={KNN_K}, PCA components={PCA_COMPONENTS}")
    print(f"Features to impute: {len(imputation_features)}")
    
    filled_df, stats_df = knn_imputation_with_pca(
        processed_data, 
        imputation_features,
        k=KNN_K,
        n_components=PCA_COMPONENTS
    )
    
    # 8. Save imputed data
    print(f"\nSaving imputed data to: {output_filled}")
    filled_df.to_csv(output_filled, index=False)
    
    print(f"Saving imputation statistics to: {output_stats}")
    stats_df.to_csv(output_stats, index=False)
    
    # 9. Final completeness check
    print("\n" + "="*50)
    print("FINAL COMPLETENESS CHECK")
    print("="*50)
    
    missing_after = filled_df[imputation_features].isna().sum().sum()
    
    if missing_after > 0:
        # Identify columns with residual missing values
        missing_cols = filled_df[imputation_features].isna().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        print(f"Warning: {missing_after} missing values remain after imputation")
        print("Columns with residual missing values:")
        for col, count in missing_cols.items()[:10]:
            print(f"  - {col}: {count} missing values")
        if len(missing_cols) > 10:
            print(f"  Showing first 10 columns, total {len(missing_cols)} columns")
        
        # Identify samples with residual missing values
        missing_samples = []
        for i, row in filled_df.iterrows():
            if row[imputation_features].isna().any():
                sample_name = filled_df.iloc[i, 0]
                missing_cols_list = row.index[row.isna()].tolist()
                missing_samples.append({
                    'Sample': sample_name,
                    'MissingColumns': len(missing_cols_list),
                    'MissingColumnNames': ", ".join(missing_cols_list[:5])
                })
        
        # Print affected samples
        print("\nAffected samples:")
        for sample_info in missing_samples[:min(5, len(missing_samples))]:
            print(f"  Sample {sample_info['Sample']}: {sample_info['MissingColumns']} columns missing")
        
        if len(missing_samples) > 5:
            print(f"  Showing first 5 samples, total affected: {len(missing_samples)}")
        
        # Save residual missing sample information
        if missing_samples:
            missing_samples_df = pd.DataFrame(missing_samples)
            missing_samples_path = "path/to/your/residual_missing_samples.csv"
            missing_samples_df.to_csv(missing_samples_path, index=False)
            print(f"Residual missing sample list saved to: {missing_samples_path}")
        
        # Fill residual missing values (optional)
        fill_method = 'median'  # Options: 'zero', 'median', 'mean'
        if fill_method == 'zero':
            filled_df[imputation_features] = filled_df[imputation_features].fillna(0)
            print("Remaining missing values filled with 0")
        elif fill_method == 'median':
            for col in missing_cols.index:
                col_median = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(col_median)
            print("Remaining missing values filled with column medians")
        else:  # 'mean'
            for col in missing_cols.index:
                col_mean = filled_df[col].mean()
                filled_df[col] = filled_df[col].fillna(col_mean)
            print("Remaining missing values filled with column means")
    else:
        print("✅ All missing values successfully imputed")
    
    # 10. Print summary statistics
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Input data: {original_shape[0]} rows, {original_shape[1]} columns")
    print(f"Processed data: {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")
    print(f"Features retained: {filtered_step3.shape[1]}")
    print(f"Features removed: {original_shape[1] - 6 - filtered_step3.shape[1]}")
    
    if 'stats_df' in locals():
        print(f"\nImputation statistics:")
        print(f"  Samples with missing values: {len(stats_df)}")
        print(f"  Average missing features per sample: {stats_df['MissingFeatures'].mean():.2f}")
        print(f"  Maximum missing features: {stats_df['MissingFeatures'].max()}")
    
    print(f"\nOutput files:")
    print(f"  Processed data: {output_processed}")
    print(f"  Imputed data: {output_filled}")
    print(f"  Imputation statistics: {output_stats}")
    
    if duplicate_report is not None:
        print(f"  Duplicate sample report: {duplicate_report_path}")

if __name__ == "__main__":
    main()