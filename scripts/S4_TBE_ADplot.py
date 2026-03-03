import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

def load_domain_data(domain_name):
    AD_DATA_DIR = 'TBEresults/ad_data'
    
    try:
        # Load metadata
        ad_data = joblib.load(f'{AD_DATA_DIR}/{domain_name}_ad_data.pkl')

        # Load feature names
        feature_names_path = f'{AD_DATA_DIR}/{domain_name}_feature_names.pkl'
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
        else:
            print(f"  Warning: Feature name file not found for {domain_name}")

        # Load training set data
        train_data = joblib.load(f'TBEresults/ad_data/{domain_name}_train_data.pkl')
        X_train = train_data['features']
        train_sample_ids = train_data['sample_ids']
        
        # Load test set data
        test_data = joblib.load(f'TBEresults/ad_data/{domain_name}_test_data.pkl')
        X_test = test_data['features']
        test_sample_ids = test_data['sample_ids']
        
        # Create DataFrame with index (optional)
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=feature_names, index=train_sample_ids)
        else:
            X_train.index = train_sample_ids
        
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=feature_names, index=test_sample_ids)
        else:
            X_test.index = test_sample_ids
        
        # Load external sample data (from prediction results)
        external_file = f'TBEresults/external_AD/{domain_name}_external.pkl'
        if not os.path.exists(external_file):
            print(f"  Warning: External sample file not found - {external_file}")
            return None
        
        external_data = joblib.load(external_file)
        X_external = external_data['features']
        external_sample_ids = external_data['sample_ids']
        external_confidences = external_data['confidences']
        feature_names = external_data['feature_names']
        
        # Convert to DataFrame
        if isinstance(X_external, np.ndarray):
            X_external = pd.DataFrame(X_external, columns=feature_names, index=external_sample_ids)
        else:
            X_external.index = external_sample_ids
        
        print(f"  Loaded external samples: {len(X_external)} samples, {len(feature_names)} features")

        # Load all-sample data (from prediction results)
        all_file = f'TBEresults/all_AD/{domain_name}_all.pkl'
        
        all_data = joblib.load(all_file)
        X_all = all_data['features']
        all_sample_ids = all_data['sample_ids']
        all_confidences = all_data['confidences']
        allfeature_names = all_data['feature_names']
        
        # Convert to DataFrame
        if isinstance(X_all, np.ndarray):
            X_all = pd.DataFrame(X_all, columns=allfeature_names, index=all_sample_ids)
        else:
            X_all.index = all_sample_ids
        
        print(f"  Loaded all samples: {len(X_all)} samples, {len(feature_names)} features")

        # Sample ID consistency check
        print("\n===== Sample ID Consistency Check =====")

        # Check training set sample IDs
        if hasattr(X_train, 'index'):
            train_sample_ids = X_train.index.tolist()
            print(f"  Training set samples: {len(train_sample_ids)}")
            print(f"  Training set sample ID examples: {train_sample_ids[:5]}")
            print(f"  Training set sample ID type: {type(train_sample_ids[0]) if train_sample_ids else 'empty'}")
        else:
            print("  Warning: Training set has no index attribute")

        # Check test set sample IDs
        if hasattr(X_test, 'index'):
            test_sample_ids = X_test.index.tolist()
            print(f"  Test set samples: {len(test_sample_ids)}")
            print(f"  Test set sample ID examples: {test_sample_ids[:5]}")
            print(f"  Test set sample ID type: {type(test_sample_ids[0]) if test_sample_ids else 'empty'}")
        else:
            print("  Warning: Test set has no index attribute")

        # Check external set sample IDs
        if hasattr(X_external, 'index'):
            external_sample_ids = X_external.index.tolist()
            print(f"  External set samples: {len(external_sample_ids)}")
            print(f"  External set sample ID examples: {external_sample_ids[:5]}")
            print(f"  External set sample ID type: {type(external_sample_ids[0]) if external_sample_ids else 'empty'}")
        else:
            print("  Warning: External set has no index attribute")

        # Check all-set sample IDs
        if hasattr(X_all, 'index'):
            all_sample_ids = X_all.index.tolist()
            print(f"  All set samples: {len(all_sample_ids)}")
            print(f"  All set sample ID examples: {all_sample_ids[:5]}")
            print(f"  All set sample ID type: {type(all_sample_ids[0]) if all_sample_ids else 'empty'}")
        else:
            print("  Warning: All set has no index attribute")

        # Check sample ID type consistency
        all_id_types = set([
            type(train_sample_ids[0]) if train_sample_ids else None,
            type(test_sample_ids[0]) if test_sample_ids else None,
            type(external_sample_ids[0]) if external_sample_ids else None,
            type(all_sample_ids[0]) if all_sample_ids else None
        ])
        
        if len(all_id_types) == 1:
            print(f"  All dataset sample ID types consistent: {list(all_id_types)[0]}")
        else:
            print(f"  Warning: Sample ID types inconsistent: {all_id_types}")

        return X_train, X_test, X_external, external_confidences, external_sample_ids, X_all, all_confidences, all_sample_ids, ad_data
    
    except FileNotFoundError as e:
        print(f"Error: Data file not found for {domain_name}: {str(e)}")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"Failed to load {domain_name} data: {str(e)}")
        return None, None, None, None, None, None, None, None

def batch_tanimoto_distance(X, Y):
    """
    Batch calculate Tanimoto distance
    X: Training set categorical feature matrix (n_train_samples, n_features)
    Y: Query set categorical feature matrix (n_query_samples, n_features)
    Returns: Distance matrix (n_query_samples, n_train_samples)
    """
    # Convert to boolean
    X = X.astype(bool)
    Y = Y.astype(bool)
    print(f"batch_tanimoto_distance: X.shape={X.shape}, Y.shape={Y.shape}, return matrix shape=({X.shape[0]}, {Y.shape[0]})")

    # Calculate intersection and union
    intersection = np.dot(Y, X.T)
    union = np.sum(Y, axis=1)[:, np.newaxis] + np.sum(X, axis=1) - intersection
    
    # Create result array (default value 1)
    similarity = np.ones_like(intersection, dtype=float)
    
    # Find positions with non-zero denominator
    valid_mask = union != 0
    
    # Divide only valid positions
    similarity[valid_mask] = intersection[valid_mask] / union[valid_mask]
    
    return 1 - similarity

def split_feature_types(features, categorical_indices=None, verbose=True, prefix=""):
    """
    Split continuous and categorical features
    :param verbose: Whether to print detailed information
    Returns: (continuous features DataFrame, categorical features DataFrame, continuous indices, categorical indices)
    """
    # Print overall feature information
    if verbose:
        print(f"{prefix}Feature splitting:")
        print(f"  Total features: {features.shape[1]}, Samples: {features.shape[0]}")

    # Use utility function to re-identify categorical features
    feature_names = features.columns.tolist()
    if categorical_indices is None:
        categorical_indices = identify_categorical_features(feature_names)
    
    # Get feature indices
    cont_indices = [i for i in range(features.shape[1]) if i not in categorical_indices]
    cat_indices = categorical_indices
    
    # Print identification results
    print(f"Identified {len(cat_indices)} categorical features:")
    if cat_indices:
        print(f"  Categorical feature indices: {cat_indices}")
        print(f"  Categorical feature names: {[feature_names[i] for i in cat_indices][:3]}...")
    
    print(f"Identified {len(cont_indices)} continuous features:")
    if cont_indices:
        print(f"  Continuous feature indices: {cont_indices[:3]}...")
        print(f"  Continuous feature names: {[feature_names[i] for i in cont_indices][:3]}...")
    
    # Split features
    cont_features = features.iloc[:, cont_indices] if cont_indices else None
    cat_features = features.iloc[:, cat_indices] if cat_indices else None
    
    # Print splitting results
    if cont_features is not None:
        print(f"Continuous features shape: {cont_features.shape}")
    if cat_features is not None:
        print(f"Categorical features shape: {cat_features.shape}")
    
    if verbose:
        print(f"{prefix}Feature splitting complete")
    
    return cont_features, cat_features, cont_indices, cat_indices

def identify_categorical_features(feature_names):
    """
    Identify categorical features
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
    
    return categorical_indices

def compute_feature_distances(features1_cont, features2_cont, features1_cat, features2_cat, n_jobs=1):
    """
    Compute mixed feature distance matrix
    Returns: (n_samples1, n_samples2) distance matrix
    """
    # Continuous feature distance
    if features1_cont is not None and features2_cont is not None:
        cont_dists = pairwise_distances(features1_cont, features2_cont, n_jobs=n_jobs)
    else:
        cont_dists = np.zeros((len(features1_cont) if features1_cont is not None else 1, 
                              len(features2_cont) if features2_cont is not None else 1))
    
    # Categorical feature distance
    if features1_cat is not None and features2_cat is not None:
        cat_dists = batch_tanimoto_distance(features2_cat.values, features1_cat.values)
    else:
        # Get sample count
        n1 = features1_cont.shape[0] if features1_cont is not None else (features1_cat.shape[0] if features1_cat is not None else 1)
        n2 = features2_cont.shape[0] if features2_cont is not None else (features2_cat.shape[0] if features2_cat is not None else 1)
        cat_dists = np.zeros((n1, n2))
    
    # After computing cont_dists and cat_dists
    if cont_dists.shape != cat_dists.shape:
        # Check if it's a transpose relationship
        raise ValueError(f"Continuous feature distance matrix shape {cont_dists.shape} doesn't match categorical feature distance matrix shape {cat_dists.shape}")
        
    # Dynamic weighting
    n_cont = features1_cont.shape[1] if features1_cont is not None else 0
    n_cat = features1_cat.shape[1] if features1_cat is not None else 0
    total_features = n_cont + n_cat
    
    if total_features > 0:
        alpha = n_cont / total_features
        return alpha * cont_dists + (1 - alpha) * cat_dists
    return cont_dists + cat_dists

def precompute_train_density(train_features, categorical_indices, n_jobs=1):
    """Efficient precomputation of training set density data"""
    print(f"Precomputing training set density data...")
    
    # Split features
    cont_train, cat_train, _, _ = split_feature_types(
        train_features, categorical_indices, 
        verbose=True, prefix="Training set"
    )

    # Compute complete distance matrix
    dists = compute_feature_distances(
        cont_train, cont_train, 
        cat_train, cat_train,
        n_jobs=n_jobs
    )

    # Set diagonal to infinity (exclude self)
    np.fill_diagonal(dists, np.inf)
    
    # Compute d_query (nearest neighbor distance)
    train_d_nn = np.min(dists, axis=1)
    train_d_nn = np.maximum(train_d_nn, 1e-8)
    
    print(f"Training set density computation complete")
    return train_d_nn

def compute_density_ratios(train_features, query_features, categorical_indices,
                          train_density_data=None, n_jobs=1):
    """Density ratio computation"""
    # Split features
    cont_train, cat_train, _, _ = split_feature_types(train_features, categorical_indices, verbose=False, prefix="Training set")
    cont_query, cat_query, _, _ = split_feature_types(query_features, categorical_indices, verbose=True, prefix="Test set")
    print(f"cont_train shape: {cont_train.shape}")
    print(f"cont_query shape: {cont_query.shape}")
    print(f"cat_train shape: {cat_train.shape if cat_train is not None else None}")
    print(f"cat_query shape: {cat_query.shape if cat_query is not None else None}")

    # Compute distance matrix
    dist_matrix = compute_feature_distances(
        cont_query, cont_train,
        cat_query, cat_train,
        n_jobs=n_jobs
    )
    
    # Check if query set and training set are the same
    if id(query_features) == id(train_features) or np.array_equal(query_features, train_features):
        print("  Warning: Query set same as training set, excluding self samples")
        # Set diagonal to infinity
        np.fill_diagonal(dist_matrix, np.inf)
    
    # Compute density ratios
    density_ratios = []
    for i in range(len(query_features)):
        # Find nearest neighbor index
        nn_idx = np.argmin(dist_matrix[i])
        
        # Use pre-stored data
        d_query = dist_matrix[i][nn_idx]
        d_nn = train_density_data[nn_idx]
        
        # Compute density ratio
        density_ratio = d_query / d_nn if d_nn > 0 else float('inf')
        density_ratios.append(density_ratio)
    
    return np.array(density_ratios)

def define_applicability_domain(domain_name, n_jobs=1):
    """Optimized applicability domain definition"""
    print(f"\n===== Processing domain: {domain_name} =====")
    # Load data
    X_train, X_test, X_external, external_confidences, external_sample_ids, X_all, all_confidences, all_sample_ids, ad_data = load_domain_data(domain_name)
    
    if X_train is None:
        print(f"  Error: Unable to load data for {domain_name}, skipping")
        return None

    # Extract categorical feature indices
    categorical_indices = ad_data.get('categorical_indices', [])
    
    # Compute training set density ratios
    print(f"\n[Training set feature splitting]")
    print(f"  Samples: {len(X_train)}, Features: {X_train.shape[1]}")
    train_density_data = precompute_train_density(
        X_train, 
        categorical_indices,
        n_jobs=n_jobs
    )
    # Save precomputed data
    ad_data['train_density_data'] = train_density_data
    joblib.dump(ad_data, f'ad_data/{domain_name}_ad_data.pkl')
    
    # Compute density ratios
    print(f"\n[Training set density computation]")
    train_density_ratios = compute_density_ratios(
        X_train, X_train, categorical_indices,
        train_density_data=train_density_data,
        n_jobs=n_jobs
    )

    # Compute threshold (95th percentile)
    threshold = np.percentile(train_density_ratios, 95)
    print(f"  Applicability domain threshold (density ratio): {threshold:.4f}")
    
    # Evaluate training set samples
    train_in_ad_density = train_density_ratios <= threshold
    train_coverage_density = np.mean(train_in_ad_density) * 100
    print(f"  Training set structural space coverage: {train_coverage_density:.1f}%")
    
    # Compute test set density ratios
    print(f"\n[Test set feature splitting]")
    print(f"  Samples: {len(X_test)}, Features: {X_test.shape[1]}")
    print(f"Computing test set density ratios ({domain_name})...")
    test_density_ratios = compute_density_ratios(
        X_train, X_test, categorical_indices,
        train_density_data=train_density_data,
        n_jobs=n_jobs
    )
    
    # Evaluate test set samples
    test_in_ad_density = test_density_ratios <= threshold
    test_coverage_density = np.mean(test_in_ad_density) * 100
    print(f"  Test set structural space coverage: {test_coverage_density:.1f}%")

    # Compute external set density ratios
    print(f"\n[External set feature splitting]")
    print(f"  Samples: {len(X_external)}, Features: {X_external.shape[1]}")
    print(f"Computing external set density ratios ({domain_name})...")
    external_density_ratios = compute_density_ratios(
        X_train, X_external, categorical_indices,
        train_density_data=train_density_data,
        n_jobs=n_jobs
    )
    
    # Evaluate external set samples
    external_in_ad_density = external_density_ratios <= threshold
    external_coverage_density = np.mean(external_in_ad_density) * 100
    print(f"  External set structural space coverage: {external_coverage_density:.1f}%")
    
    # Compute all-sample density ratios
    print(f"\n[All set feature splitting]")
    print(f"  Samples: {len(X_all)}, Features: {X_all.shape[1]}")
    print(f"Computing all set density ratios ({domain_name})...")
    all_density_ratios = compute_density_ratios(
        X_train, X_all, categorical_indices,
        train_density_data=train_density_data,
        n_jobs=n_jobs
    )
    
    # Evaluate all set samples
    all_in_ad_density = all_density_ratios <= threshold
    all_coverage_density = np.mean(all_in_ad_density) * 100
    print(f"  All set structural space coverage: {all_coverage_density:.1f}%")
    
    # Extract confidence data
    train_confidence = ad_data.get('train_confidence', None)
    test_confidence = ad_data.get('test_confidence', None)
    
    confidence_threshold = 0.525
    print(f"  Confidence applicability domain threshold: {confidence_threshold:.2f}")
    
    # Evaluate training set samples
    if train_confidence is not None:
        train_in_ad_confidence = train_confidence >= confidence_threshold
        train_coverage_confidence = np.mean(train_in_ad_confidence) * 100
        print(f"  Training set confidence coverage: {train_coverage_confidence:.1f}%")
    else:
        print("  Warning: No training set confidence data")
        train_in_ad_confidence = np.ones(len(X_train), dtype=bool)
        train_coverage_confidence = 100.0
    
    # Evaluate test set samples
    if test_confidence is not None:
        test_in_ad_confidence = test_confidence >= confidence_threshold
        test_coverage_confidence = np.mean(test_in_ad_confidence) * 100
        print(f"  Test set confidence coverage: {test_coverage_confidence:.1f}%")
    else:
        print("  Warning: No test set confidence data")
        test_in_ad_confidence = np.ones(len(X_test), dtype=bool)
        test_coverage_confidence = 100.0
    
    # Evaluate external set samples
    if external_confidences is not None:
        external_in_ad_confidence = external_confidences >= confidence_threshold
        external_coverage_confidence = np.mean(external_in_ad_confidence) * 100
        print(f"  External set confidence coverage: {external_coverage_confidence:.1f}%")
    else:
        print("  Warning: No external set confidence data")
        external_in_ad_confidence = np.ones(len(X_external), dtype=bool)
        external_coverage_confidence = 100.0

    # Evaluate all set samples
    if all_confidences is not None:
        all_in_ad_confidence = all_confidences >= confidence_threshold
        all_coverage_confidence = np.mean(all_in_ad_confidence) * 100
        print(f"  All set confidence coverage: {all_coverage_confidence:.1f}%")
    else:
        print("  Warning: No all set confidence data")
        all_in_ad_confidence = np.ones(len(X_all), dtype=bool)
        all_coverage_confidence = 100.0

    # Compute samples satisfying both thresholds
    train_mask_in_high = (train_density_ratios <= threshold) & (train_confidence >= confidence_threshold)
    test_mask_in_high = (test_density_ratios <= threshold) & (test_confidence >= confidence_threshold)
    external_mask_in_high = (external_density_ratios <= threshold) & (external_confidences >= confidence_threshold)

    train_coverage_combined = np.mean(train_mask_in_high) * 100
    test_coverage_combined = np.mean(test_mask_in_high) * 100
    external_coverage_combined = np.mean(external_mask_in_high) * 100

    all_mask_in_high = (all_density_ratios <= threshold) & (all_confidences >= confidence_threshold)
    all_coverage_combined = np.mean(all_mask_in_high) * 100
    
    # Save results
    results = {
        'domain': domain_name,
        'threshold': threshold,
        'confidence_threshold': confidence_threshold,
        'train_density_ratios': train_density_ratios,
        'test_density_ratios': test_density_ratios,
        'external_density_ratios': external_density_ratios,
        'train_confidence': train_confidence,
        'test_confidence': test_confidence,
        'external_confidence': external_confidences,
        'train_in_ad_density': train_in_ad_density,
        'test_in_ad_density': test_in_ad_density,
        'external_in_ad_density': external_in_ad_density,
        'train_in_ad_confidence': train_in_ad_confidence,
        'test_in_ad_confidence': test_in_ad_confidence,
        'external_in_ad_confidence': external_in_ad_confidence,
        'train_coverage_density': train_coverage_density,
        'test_coverage_density': test_coverage_density,
        'external_coverage_density': external_coverage_density,
        'train_coverage_confidence': train_coverage_confidence,
        'test_coverage_confidence': test_coverage_confidence,
        'external_coverage_confidence': external_coverage_confidence,
        'train_coverage_combined': train_coverage_combined,
        'test_coverage_combined': test_coverage_combined,
        'external_coverage_combined': external_coverage_combined,
        'all_coverage_density': all_coverage_density,
        'all_coverage_confidence': all_coverage_confidence,
        'all_coverage_combined': all_coverage_combined
    }
      
    # Debug: print array lengths
    print(f"\n[Debug] Checking array lengths before saving:")
    print(f"  Training set samples: {len(X_train)}")
    print(f"  train_density_ratios length: {len(results['train_density_ratios'])}")
    print(f"  train_confidence length: {len(results['train_confidence']) if results['train_confidence'] is not None else 'None'}")
    print(f"  train_in_ad_density length: {len(results['train_in_ad_density'])}")
    print(f"  train_in_ad_confidence length: {len(results['train_in_ad_confidence'])}")
    
    print(f"\n  Test set samples: {len(X_test)}")
    print(f"  test_density_ratios length: {len(results['test_density_ratios'])}")
    print(f"  test_confidence length: {len(results['test_confidence']) if results['test_confidence'] is not None else 'None'}")
    print(f"  test_in_ad_density length: {len(results['test_in_ad_density'])}")
    print(f"  test_in_ad_confidence length: {len(results['test_in_ad_confidence'])}")

    # Create results directory
    ad_output = f'TBEad_results/{domain_name}'
    os.makedirs(ad_output, exist_ok=True)
    
    # Save results file
    save_domain_results(results, X_train, X_test, external_sample_ids)
    
    # Visualization
    visualize_domain_coverage(results, domain_name)
    
    print(f"\n===== Completed processing domain: {domain_name} =====")
    return results

def save_domain_results(results, X_train, X_test, external_sample_ids):
    """Save domain results"""
    domain_name = results['domain']
    
    # Save result object
    joblib.dump(results, f'TBEad_results/{domain_name}/{domain_name}_ad_results.pkl')

    # Save CSV files
    train_df = pd.DataFrame({
        'Compound_CID': X_train.index,
        'density_ratio': results['train_density_ratios'],
        'confidence': results['train_confidence'] if results['train_confidence'] is not None else np.nan,
        'in_ad_density': results['train_in_ad_density'],
        'in_ad_confidence': results['train_in_ad_confidence']
    })
    
    test_df = pd.DataFrame({
        'Compound_CID': X_test.index,
        'density_ratio': results['test_density_ratios'],
        'confidence': results['test_confidence'] if results['test_confidence'] is not None else np.nan,
        'in_ad_density': results['test_in_ad_density'],
        'in_ad_confidence': results['test_in_ad_confidence']
    })

    # Save external sample results
    if results['external_density_ratios'] is not None and results['external_confidence'] is not None:
        external_df = pd.DataFrame({
            'Compound_CID': external_sample_ids,
            'density_ratio': results['external_density_ratios'],
            'confidence': results['external_confidence'],
            'in_ad_density': results['external_in_ad_density'],
            'in_ad_confidence': results['external_in_ad_confidence']
        })
        external_df.to_csv(f'TBEad_results/{domain_name}/{domain_name}_external_ad.csv', index=False)
    
    train_df.to_csv(f'TBEad_results/{domain_name}/{domain_name}_train_ad.csv', index=False)
    test_df.to_csv(f'TBEad_results/{domain_name}/{domain_name}_test_ad.csv', index=False)

def visualize_domain_coverage(results, domain_name):
    """Visualize applicability domain coverage"""
    # Create 1 row, 2 column subplot layout
    fig, (ax_main, ax_outlier) = plt.subplots(1, 2, figsize=(15, 8), dpi=100,
                                              gridspec_kw={'width_ratios': [3, 1], 'wspace': 0})
    
    # Set main title
    plt.suptitle(f'Domain-Specific TBE - {domain_name}', fontsize=15, y=0.92)
    # Extract data
    train_density = np.array(results['train_density_ratios'])
    train_confidence = np.array(results['train_confidence'])
    test_density = np.array(results['test_density_ratios'])
    test_confidence = np.array(results['test_confidence'])
    external_density = np.array(results['external_density_ratios'])
    external_confidence = np.array(results['external_confidence'])
    density_threshold = results['threshold']
    confidence_threshold = results['confidence_threshold']
    
    # Calculate quadrant proportions
    train_mask_in_high = (train_density <= density_threshold) & (train_confidence > confidence_threshold)
    test_mask_in_high = (test_density <= density_threshold) & (test_confidence > confidence_threshold)
    external_mask_in_high = (external_density <= density_threshold) & (external_confidence > confidence_threshold)

    # Calculate proportions
    train_high_in = f"{len(train_density[train_mask_in_high]) / len(train_density) * 100:.1f}%"
    test_high_in = f"{len(test_density[test_mask_in_high]) / len(test_density) * 100:.1f}%"
    external_high_in = f"{len(external_density[external_mask_in_high]) / len(external_density) * 100:.1f}%"
    
    # Main plot
    x_limit = density_threshold + 0.01
    
    # Filter main plot region points
    mask_train_main = train_density <= x_limit
    mask_test_main = test_density <= x_limit
    mask_external_main = external_density <= x_limit
    
    # Plot external set samples
    ax_main.scatter(
        external_density[mask_external_main], 
        external_confidence[mask_external_main],
        marker='d', s=20, facecolor='none', edgecolor='#808080', alpha=0.6, label='External Set'
    )

    # Plot training set samples
    ax_main.scatter(
        train_density[mask_train_main], 
        train_confidence[mask_train_main],
        alpha=1, c='#FFC208', marker='o', s=25, label='Training Set'
    )
    
    # Plot test set samples
    ax_main.scatter(
        test_density[mask_test_main], 
        test_confidence[mask_test_main],
        alpha=1, c='#073A94', marker='^', s=30, label='Test Set'
    )
        
    # Plot threshold lines
    ax_main.plot(
        [density_threshold, density_threshold], 
        [0.49, 1.01], 
        color='#be1420',  
        linestyle='--', 
        linewidth=1.5,
        label=f'Density Threshold ({density_threshold:.2f})'
    )
    ax_main.axhline(
        y=confidence_threshold, 
        color='#be1420',  
        linestyle='-.', 
        linewidth=1.5,
        label=f'Confidence Threshold ({confidence_threshold:.2f})'
    )
    
    # Main plot settings
    ax_main.text(
        x= density_threshold / 1.5,
        y= 0.46,
        s='Local Density Ratio',
        fontsize=13,
        ha='center',
        va='top',
    )
    ax_main.set_ylabel('Confidence', fontsize=13)
    
    # Adjust range based on main plot data
    main_data_max = max(np.max(train_density[mask_train_main]), 
                        np.max(test_density[mask_test_main]), 
                        np.max(external_density[mask_external_main]))
    ax_main.set_xlim(-0.1, min(main_data_max * 1.2, x_limit))
    ax_main.set_ylim(0.488, 1.01)
    ax_main.spines['bottom'].set_position(('data', 0.49))
    
    # Set main plot ticks
    x_min, x_max = ax_main.get_xlim()
    if x_max <= 1.0:
        step = 0.1
    elif x_max <= 5.0:
        step = 0.5
    else:
        step = 1.0
        
    major_ticks = np.arange(0, np.ceil(x_max), step)
    major_ticks = major_ticks[major_ticks < density_threshold]
    ax_main.set_xticks(major_ticks)
    ax_main.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_main.tick_params(axis='both', which='major', labelsize=12)
    
    # Add quadrant labels
    quad_bbox = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax_main.text(density_threshold * 0.95 , confidence_threshold + 0.03, 
                 f"Coverage in AD    \nTraining Set: {train_high_in}\nTest Set: {test_high_in}\nExternal Set: {external_high_in}", 
                 ha='right', va='bottom', fontsize=12, bbox=quad_bbox)

    ax_main.spines['right'].set_visible(False)
    
    # Outlier plot
    train_mask_outlier = train_density > density_threshold
    test_mask_outlier = test_density > density_threshold
    external_mask_outlier = external_density > density_threshold
    train_outliers_count = sum(train_mask_outlier)
    test_outliers_count = sum(test_mask_outlier)
    external_outliers_count = sum(external_mask_outlier)
    total_outliers = train_outliers_count + test_outliers_count + external_outliers_count
    
    # Add confidence threshold line
    ax_outlier.axhline(
        y=confidence_threshold, 
        color='#be1420',
        linestyle='-.', 
        linewidth=1.5
    )
    
    if total_outliers > 0:
        # Collect all outliers
        all_outliers = {
            'density': np.concatenate([train_density[train_mask_outlier], test_density[test_mask_outlier], external_density[external_mask_outlier]]),
            'confidence': np.concatenate([train_confidence[train_mask_outlier], test_confidence[test_mask_outlier], external_confidence[external_mask_outlier]]),
            'type': np.concatenate([
                ['train'] * train_outliers_count, 
                ['test'] * test_outliers_count, 
                ['external'] * external_outliers_count
            ])
        }
        
        # Sort outliers by density value
        sorted_idx = np.argsort(all_outliers['density'])
        density_sorted = all_outliers['density'][sorted_idx]
        confidence_sorted = all_outliers['confidence'][sorted_idx]
        type_sorted = all_outliers['type'][sorted_idx]
        
        # Create position coordinates
        positions = np.arange(1, total_outliers + 1)
        
        # Create position offsets to avoid overlap
        offsets = np.linspace(-0.1, 0.1, total_outliers)
                
        # Plot external set outliers
        external_indices = [i for i, t in enumerate(type_sorted) if t == 'external']
        if external_indices:
            ax_outlier.scatter(
                positions[external_indices] + offsets[external_indices],
                confidence_sorted[external_indices],
                marker='d', s=20, facecolor='none', edgecolor='#808080', alpha=0.6
            )
         
        # Plot training set outliers
        train_indices = [i for i, t in enumerate(type_sorted) if t == 'train']
        if train_indices:
            ax_outlier.scatter(
                positions[train_indices] + offsets[train_indices],
                confidence_sorted[train_indices],
                alpha=1, c='#FFC208', marker='o', s=25
            )
        
        # Plot test set outliers
        test_indices = [i for i, t in enumerate(type_sorted) if t == 'test']
        if test_indices:
            ax_outlier.scatter(
                positions[test_indices] + offsets[test_indices],
                confidence_sorted[test_indices],
                alpha=1, c='#073A94', marker='^', s=30
            )
   
        # Calculate min and max values
        min_value = np.min(density_sorted)
        max_value = np.max(density_sorted)
        max_str = f"{max_value:.2e}" if max_value > 1000 else f"{max_value:.2f}"
        
        # Add range label
        range_text = f"{min_value:.2f} ~ {max_str}\n(Outliers Range)"
        ax_outlier.text( total_outliers/2 + 0.3, 0.482, range_text, 
                       ha='center', va='top', fontsize=12, 
                       transform=ax_outlier.transData)
        
        # Draw wavy line
        x_min, x_max = 0.5, total_outliers + 0.5
        y_min, y_mid, y_max = 0.488, 0.490, 0.492

        amplitude = (y_max - y_min) / 2
        cycles = 20
        frequency = cycles * 2 * np.pi / (x_max - x_min)

        x = np.linspace(x_min, x_max, 1000)
        y = amplitude * np.sin(frequency * (x - x_min)) + y_mid

        ax_outlier.plot(x, y, linestyle='-', color='black', linewidth= 1.0)

    # Outlier plot settings
    ax_outlier.set_ylim(0.488, 1.01)
    ax_outlier.set_xlim(0.5, total_outliers + 0.5 if total_outliers > 0 else 1.5)
    
    # Hide all ticks
    ax_outlier.set_xticks([])
    ax_outlier.set_yticks([])
    ax_outlier.set_yticklabels([])
    ax_outlier.spines['bottom'].set_visible(False)
    ax_outlier.spines['left'].set_visible(False)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='#FFC208', alpha=1, marker='o', linestyle='None', markersize=8, label='Training Set'),
        Line2D([0], [0], color='#073A94', alpha=1, marker='^', linestyle='None', markersize=8, label='Test Set'),
        Line2D([0], [0], marker='d', markeredgecolor='#808080', markerfacecolor='none', linestyle='None', markersize=8, label='External Set'),
        Line2D([0], [0], color='#be1420', linestyle='--', linewidth=1.5, label=f'ρ Threshold ({density_threshold:.2f})'),
        Line2D([0], [0], color='#be1420', linestyle='-.', linewidth=1.5, label=f'Conf. Threshold ({confidence_threshold:.3f})')
    ]
    
    ax_outlier.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.5)
    
    # Adjust layout and save
    os.makedirs('TBEad_results', exist_ok=True)
    plt.savefig(f'TBEad_results/{domain_name}/{domain_name}_domain_TBE.png', dpi=300, bbox_inches='tight')
    plt.close()

def batch_process_domains(domain_list, max_workers=64):
    """Parallel processing at domain level"""
    num_domains = len(domain_list)
    n_jobs_per_domain = max(1, max_workers // num_domains) if num_domains > 0 else 1
    
    if num_domains > max_workers:
        n_jobs_per_domain = 1
    else:
        n_jobs_per_domain = max(1, max_workers // num_domains)
    
    print(f"Domain-level parallel processing: {num_domains} domains, {n_jobs_per_domain} cores per domain")
    
    # Parallel processing for each domain
    results = Parallel(n_jobs=max_workers)(
        delayed(process_domain)(domain_name, n_jobs=n_jobs_per_domain)
        for domain_name in tqdm(domain_list, desc="Processing domains in parallel")
    )
    
    # Integrate valid results
    domain_results = {}
    coverage_data = []

    for result in results:
        if result is not None:
            domain_name = result['domain']
            domain_results[domain_name] = result

            coverage_data.append({
                'Domain': domain_name,
                'Train_Coverage_Combined': result['train_coverage_combined'],
                'Test_Coverage_Combined': result['test_coverage_combined'],
                'External_Coverage_Combined': result['external_coverage_combined'],
                'Train_Coverage_Density': result['train_coverage_density'],
                'Test_Coverage_Density': result['test_coverage_density'],
                'External_Coverage_Density': result['external_coverage_density'],
                'Train_Coverage_Confidence': result['train_coverage_confidence'],
                'Test_Coverage_Confidence': result['test_coverage_confidence'],
                'External_Coverage_Confidence': result['external_coverage_confidence'],
                'All_Coverage_Combined': result['all_coverage_combined'],
                'All_Coverage_Density': result['all_coverage_density'],
                'All_Coverage_Confidence': result['all_coverage_confidence']
            })

    # Save coverage data as CSV
    if coverage_data:
        coverage_df = pd.DataFrame(coverage_data)
        coverage_df.to_csv('TBEad_results/domain_coverage_summary.csv', index=False)
        print(f"Coverage summary saved to: TBEad_results/domain_coverage_summary.csv")

    return domain_results

def process_domain(domain_name, n_jobs=1):
    """Single domain processing function"""
    print(f"================================Starting domain: {domain_name}")
    try:
        result = define_applicability_domain(domain_name, n_jobs=n_jobs)
        print(f"Completed domain================================: {domain_name}")
        return result
    except Exception as e:
        print(f"  Error: {domain_name} processing failed - {str(e)}")
        return None

def main():
    """Main function with parallel control support"""
    import argparse
    parser = argparse.ArgumentParser(description='Applicability domain calculation')
    parser.add_argument('--domain', help='Single domain name')
    parser.add_argument('--domain_list', help='Domain list file')
    parser.add_argument('--all_domains', action='store_true', help='Process all available domains')
    parser.add_argument('--workers', type=int, default=64, help='Parallel worker count (default: 64)')
    args = parser.parse_args()
    
    domains = []
    
    if args.domain:
        domains = [args.domain]
    elif args.domain_list:
        try:
            with open(args.domain_list) as f:
                domains = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Failed to load domain list: {str(e)}")
            return
    elif args.all_domains:
        domains = discover_domains()
    
    if domains:
        print(f"Starting processing of {len(domains)} domains")
        batch_process_domains(domains)
    else:
        print("Please specify processing mode: --domain, --domain_list or --all_domains")

def discover_domains():
    """Automatically discover available domains"""
    data_dir = 'TBEresults/ad_data'
    if os.path.exists(data_dir):
        return [f.replace('_ad_data.pkl', '') 
                for f in os.listdir(data_dir) 
                if f.endswith('_ad_data.pkl')]
    return []

if __name__ == "__main__":
    main()