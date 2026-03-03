import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

# Set plotting style and fonts
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['axes.titleweight'] = 'bold'

# Model names and corresponding data directories
MODELS = {
    'MLP': {
        'roc_dir': 'evaluation_data/MLP',
        'pr_dir': 'evaluation_data/MLP',
        'auc_file': 'evaluation_data/MLP/auc_summary.csv',
        'metrics_file': 'MLPresults/test_domain_metrics.csv',
        'color': '#df7e21',
        'bar_color': '#df7e2160'
    },
    'TBE': {
        'roc_dir': 'evaluation_data/TBE',
        'pr_dir': 'evaluation_data/TBE',
        'auc_file': 'evaluation_data/TBE/auc_summary.csv',
        'metrics_file': 'TBEresults/domain_model_metrics.csv',  # New
        'color': '#922935',
        'bar_color': '#92293560'
    }
}

# Create output directories
os.makedirs('model_comparison/roc_curves', exist_ok=True)

def load_curve_data(domain, model_name):
    """Load ROC data for specified domain and model"""
    data = {
        'roc': None,
        'auc': None
    }
    
    model_data = MODELS[model_name]
    
    # Load ROC curve data
    roc_file = f"{model_data['roc_dir']}/{domain}_roc.csv"
    if os.path.exists(roc_file):
        roc_df = pd.read_csv(roc_file)
        data['roc'] = {
            'x': roc_df['fpr'],
            'y': roc_df['tpr']
        }
    
    # Load AUC values
    if os.path.exists(model_data['auc_file']):
        auc_df = pd.read_csv(model_data['auc_file'])
        domain_auc = auc_df[auc_df['domain'] == domain]
        if not domain_auc.empty:
            data['auc'] = {
                'roc_auc': domain_auc['roc_auc'].values[0],
                'baseline': domain_auc['baseline'].values[0] if 'baseline' in domain_auc else None
            }
    
    return data

def get_domains():
    """Get list of all available domain names"""
    mlp_dir = MODELS['MLP']['roc_dir']
    all_files = os.listdir(mlp_dir)
    domains = []
    
    for file in all_files:
        if file.endswith('_roc.csv'):
            domain = file.replace('_roc.csv', '')
            # Check if this domain has data in TBE model
            tbe_file = f"{MODELS['TBE']['roc_dir']}/{domain}_roc.csv"
            if os.path.exists(tbe_file):
                domains.append(domain)
    
    return domains

def load_metrics_data(model_name):
    """Load test set metric data for specified model and print detailed information"""
    metrics_file = MODELS[model_name]['metrics_file']
    
    # Check if file exists
    print(f"\n{'='*60}")
    print(f"Loading {model_name} model metric data")
    print(f"File path: {metrics_file}")
    
    if not os.path.exists(metrics_file):
        print(f"Error: File does not exist!")
        return None
    
    try:
        # Read CSV file
        df = pd.read_csv(metrics_file)
        print(f"✅ File read successfully! Found {len(df)} rows, {len(df.columns)} columns")
        
        # Print original column names
        print(f"\nOriginal column names ({model_name}):")
        print(', '.join(df.columns.tolist()))
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Define column name mapping
    column_mapping = {
        'accuracy': ['accuracy', 'test_accuracy'],
        'recall': ['recall', 'test_recall', 'sensitivity', 'test_sensitivity'],
        'specificity': ['specificity', 'test_specificity'],
        'f1_score': ['f1_score', 'test_f1_score'],
        'auc_roc': ['auc_roc', 'test_auc_roc'],
        'mcc': ['mcc', 'test_mcc'],
        'domain': ['domain']
    }
    
    # Apply column name mapping
    mapped_columns = {}
    for standard_name, variants in column_mapping.items():
        for variant in variants:
            if variant in df.columns:
                df.rename(columns={variant: standard_name}, inplace=True)
                if standard_name not in mapped_columns:
                    mapped_columns[standard_name] = variant
                break
    
    # Print mapping results
    print(f"\nColumn name mapping results ({model_name}):")
    for standard, original in mapped_columns.items():
        print(f"{original} → {standard}")
    
    # Check if required columns exist
    required_columns = ['domain', 'accuracy', 'recall', 'specificity', 'f1_score', 'auc_roc', 'mcc']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {', '.join(missing_columns)}")
        # Temporarily fill missing columns with 0
        for col in missing_columns:
            if col not in df.columns:
                df[col] = 0.0
    else:
        print(f"All required columns exist")
    
    # Ensure correct data types
    for col in required_columns[1:]:  # Skip domain column
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"{col} data type: numeric ({df[col].dtype})")
        else:
            print(f"{col} data type: {df[col].dtype}, attempting to convert to numeric")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Remove overall result rows
    overall_identifiers = ['OVERALL', 'All', 'Total', 'Summary']
    overall_mask = df['domain'].isin(overall_identifiers)
    print(f"\nRemoving overall result rows: {df[overall_mask]['domain'].tolist()}")
    df = df[~overall_mask]
    
    # Print detailed domain metric information
    print(f"\n{model_name} model 24 domains detailed metric values:")
    print("-" * 90)
    print(f"{'Domain':<10} | {'Accuracy':>8} | {'Recall':>8} | {'Specificity':>8} | {'F1 Score':>8} | {'AUC ROC':>8} | {'MCC':>8}")
    print("-" * 90)
    
    # Ensure at least 24 domains are displayed
    display_df = df.head(24) if len(df) > 24 else df
    
    for _, row in display_df.iterrows():
        print(f"{row['domain']:<10} | "
              f"{row['accuracy']:>8.4f} | "
              f"{row['recall']:>8.4f} | "
              f"{row['specificity']:>8.4f} | "
              f"{row['f1_score']:>8.4f} | "
              f"{row['auc_roc']:>8.4f} | "
              f"{row['mcc']:>8.4f}")
    
    print(f"\nProcessed data for {len(display_df)} domains")
    print('=' * 60)
    
    # Return only required columns
    return df[required_columns]

def plot_roc_comparison(domain, mlp_data, tbe_data, mlp_metrics, tbe_metrics):
    """Plot ROC curve comparison, add 3D bar chart in lower right corner"""
    # Create square figure
    fig = plt.figure(figsize=(8, 8))
    
    # === Part 1: ROC curve main plot ===
    ax1 = fig.add_axes([0.15, 0.15, 0.85, 0.85])  # Square main plot area
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    ax1.patch.set_alpha(0)
    ax1.set_zorder(3) # Set higher zorder

    # Set axis range
    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-0.01, 1.01)
    ax1.set_aspect('equal')  # Ensure equal axis proportion
     
    # === Move x-axis to top ===
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
            
    # Plot MLP model ROC curve
    if mlp_data['roc'] and mlp_data['auc']:
        ax1.plot(mlp_data['roc']['x'], mlp_data['roc']['y'], 
               color=MODELS['MLP']['color'], 
               lw=2.5, alpha=0.9, label=f"MLP")
        
    # Plot TBE model ROC curve
    if tbe_data['roc'] and tbe_data['auc']:
        ax1.plot(tbe_data['roc']['x'], tbe_data['roc']['y'], 
               color=MODELS['TBE']['color'], 
               lw=2.5, alpha=0.9, label=f"TBE")
    
    # === Main plot style adjustment ===
    # Axis label settings
    # Domain title (upper left corner)
    ax1.text(0.02, 0.98, domain, transform=ax1.transAxes, 
             fontsize=22, fontweight='bold', ha='left', va='top')
    # Place legend below domain title, arranged in two rows
    ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.93), ncol=1, prop={'size': 16, 'weight': 'bold'}, frameon=True, framealpha=0.8, handlelength=1.0)
    # Axis label settings
    ax1.set_xlabel('False Positive Rate', fontsize=20, fontweight='bold', labelpad=15)
    ax1.set_ylabel('True Positive Rate', fontsize=20, fontweight='bold', labelpad=15)
    
    # Tick settings: major ticks inward, larger font
    ax1.xaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.tick_params(axis='x', which='major', direction='out', length=8, labelsize=18, top=True, bottom=False)
    ax1.tick_params(axis='y', which='major', direction='out', length=8, labelsize=18, left=True, right=False)

    # Remove grid lines
    ax1.grid(False)
    
    # === Part 2: 3D bar chart ===
    # Metric list (new order)
    metrics = ['mcc', 'f1_score', 'auc_roc', 'recall', 'specificity', 'accuracy']
    metric_labels = ['MCC', 'F1 score', 'AUC', 'Recall', 'Specificity', 'Accuracy']
    
    # Do not convert to percentage, keep original values
    mlp_values = [mlp_metrics.get(m, 0) for m in metrics]
    tbe_values = [tbe_metrics.get(m, 0) for m in metrics]
    
    # Create 3D coordinate axis
    ax2 = fig.add_axes([0.38, 0.15, 0.6, 0.51], projection='3d')  # Adjust position and size
    ax2.set_zorder(2) # Set lower zorder

    # Set transparent background
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    # Set grid lines (light gray)
    ax2.grid(True, color='#f0f0f0', linestyle='-', linewidth=0.5)
    
    # Make axis lines and ticks visible
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    
    # Set coordinate positions (optimize x/y axis proportion)
    xpos = np.array([0, 1])  # MLP and TBE
    ypos = np.arange(len(metrics))
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(len(xpos))
    
    # Set bar dimensions (adjusted to square, shorten x-axis)
    dx = 1.0 * np.ones(len(xpos))  # Narrower in X direction
    dy = 0.8 * np.ones(len(xpos))  # Wider in Y direction
    dz = np.array([val for pair in zip(mlp_values, tbe_values) for val in pair])  # Height values
    
    # Color settings - 50% opacity
    mlp_color = MODELS['MLP']['bar_color']
    tbe_color = MODELS['TBE']['bar_color']

    # Draw 3D bar chart
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, 
             color=[mlp_color if x == 0 else tbe_color for x in xpos],
             shade=True, edgecolor='white', linewidth=0.3)

    # === 3D plot style adjustment ===
    # X-axis settings
    ax2.set_xticks([0.5, 1.5])  # Model positions
    ax2.set_xticklabels(['MLP', 'TBE'], fontsize=14, fontweight='bold', ha='right', rotation=-33)
    ax2.tick_params(axis='x', pad=1)
    ax2.set_xlim(0, 2)  # x-axis range
    
    # Y-axis settings (lengthen y-axis)
    ax2.set_yticks(np.arange(len(metrics))+0.4)
    ax2.set_yticklabels(metric_labels, fontsize=14, fontweight='bold', ha='left', rotation=-33)
    ax2.tick_params(axis='y', pad=1)
    ax2.set_ylim(0, len(metrics))
    
    # Z-axis settings
    ax2.set_zlim(0, 1)
    ax2.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_zticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=12, fontweight='bold', ha='left')
    ax2.tick_params(axis='z', pad=1)

    ax2.set_box_aspect([3, 8, 3])   # Set display length for each axis
    
    # Value labels - parallel to top surface
    for i in range(len(xpos)):
        if xpos[i] == 0:
            color = '#333333'
        else:
            color = '#D5D5D5'
        
        ax2.text(xpos[i] + dx[i]/2, 
                ypos[i] + dy[i]/2, 
                dz[i] + 0.005, 
                f'{dz[i]*100:.1f}',
                color=color, 
                fontsize=14,
                fontweight = 'bold', 
                ha='center', 
                va='bottom',
                zorder=100)
    
    # Set viewing angle
    ax2.view_init(elev=30, azim=-45)

    # Draw random guess line (diagonal)
    ax1.plot([0, 1], [0, 1], color='#696969', lw=1.5, linestyle='--', alpha=0.7)
        
    # Save image
    save_path = f'model_comparison/roc_curves/{domain}_roc_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    # Get all available domain names
    domains = get_domains()
    print(f"Found {len(domains)} domains available for comparison")
    
    # Load MLP and TBE metric data
    mlp_metrics_df = load_metrics_data('MLP')
    tbe_metrics_df = load_metrics_data('TBE')
    
    # Exit if loading fails
    if mlp_metrics_df is None or tbe_metrics_df is None:
        print("Error: Unable to load metric data, please check previous error messages")
        return
    
    for domain in domains:
        print(f"\nProcessing domain: {domain}")
        
        # Load MLP model data
        mlp_data = load_curve_data(domain, 'MLP')
        if mlp_data and mlp_data['auc']:
            print(f"  MLP data: ROC AUC={mlp_data['auc']['roc_auc']:.4f}")
        else:
            print("  MLP data: No data available")
        
        # Load TBE model data
        tbe_data = load_curve_data(domain, 'TBE')
        if tbe_data and tbe_data['auc']:
            print(f"  TBE data: ROC AUC={tbe_data['auc']['roc_auc']:.4f}")
        else:
            print("  TBE data: No data available")
        
        # Extract test metrics for this domain in both models
        mlp_domain_metrics = mlp_metrics_df[mlp_metrics_df['domain'] == domain].iloc[0].to_dict()
        tbe_domain_metrics = tbe_metrics_df[tbe_metrics_df['domain'] == domain].iloc[0].to_dict()
        
        # Plot ROC comparison (including 3D bar chart)
        roc_path = plot_roc_comparison(domain, mlp_data, tbe_data, mlp_domain_metrics, tbe_domain_metrics)
        print(f"  ROC comparison chart saved to: {roc_path}")
    
    print("\nAll domains processed!")

if __name__ == "__main__":
    main()