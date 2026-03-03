import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# File paths
tbe_file = 'path/to/TBEad_results/domain_coverage_summary.csv'
mlp_file = 'path/to/MLPad_results/domain_coverage_summary.csv'
tbe_metrics_file = 'path/to/TBEresults/domain_model_metrics.csv'
mlp_metrics_file = 'path/to/MLPresults/test_domain_metrics.csv'

# Read data
df_tbe = pd.read_csv(tbe_file)
df_mlp = pd.read_csv(mlp_file)
df_tbe_metrics = pd.read_csv(tbe_metrics_file)
df_mlp_metrics = pd.read_csv(mlp_metrics_file)

# Custom domain order
domain_order = [
    'A1', 'A2',
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
    'C1', 'C2', 'C3', 'C4',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11'
]

# Extract domain identifier
def extract_domain_id(name):
    match = re.search(r'\[(.*?)\]', name)
    return match.group(1) if match else name

# Apply processing function
df_tbe['Domain_ID'] = df_tbe['Domain'].apply(extract_domain_id)
df_mlp['Domain_ID'] = df_mlp['Domain'].apply(extract_domain_id)
df_tbe_metrics['Domain_ID'] = df_tbe_metrics['domain'].apply(extract_domain_id)
df_mlp_metrics['Domain_ID'] = df_mlp_metrics['domain'].apply(extract_domain_id)

# Ensure sorting by custom order
df_tbe['Domain_ID'] = pd.Categorical(df_tbe['Domain_ID'], categories=domain_order, ordered=True)
df_mlp['Domain_ID'] = pd.Categorical(df_mlp['Domain_ID'], categories=domain_order, ordered=True)
df_tbe_metrics['Domain_ID'] = pd.Categorical(df_tbe_metrics['Domain_ID'], categories=domain_order, ordered=True)
df_mlp_metrics['Domain_ID'] = pd.Categorical(df_mlp_metrics['Domain_ID'], categories=domain_order, ordered=True)

df_tbe = df_tbe.sort_values('Domain_ID').reset_index(drop=True)
df_mlp = df_mlp.sort_values('Domain_ID').reset_index(drop=True)
df_tbe_metrics = df_tbe_metrics.sort_values('Domain_ID').reset_index(drop=True)
df_mlp_metrics = df_mlp_metrics.sort_values('Domain_ID').reset_index(drop=True)

top_columns = [
    'Train_Coverage_Density', 'Train_Coverage_Combined',
    'Test_Coverage_Density', 'Test_Coverage_Combined',
    'External_Coverage_Density', 'External_Coverage_Combined'
]

top_column_order = []
for col in top_columns:
    top_column_order.append((col, 'TBE'))
    top_column_order.append((col, 'MLP'))

bottom_columns = ['All_Coverage_Density', 'All_Coverage_Combined']
bottom_column_order = []
for col in bottom_columns:
    bottom_column_order.append((col, 'TBE'))
    bottom_column_order.append((col, 'MLP'))

bottom_column_order.append(('All_Coverage_Combined', 'TBE-Recall'))
bottom_column_order.append(('All_Coverage_Combined', 'MLP-Recall'))

top_heatmap_data = []
bottom_heatmap_data = []

for i, (tbe_row, mlp_row) in enumerate(zip(df_tbe.itertuples(), df_mlp.itertuples())):
    top_row_data = []
    for col, source in top_column_order:
        if source == 'TBE':
            top_row_data.append(getattr(tbe_row, col))
        else:
            top_row_data.append(getattr(mlp_row, col))
    top_heatmap_data.append(top_row_data)
    
    bottom_row_data = []
    for col, source in bottom_column_order:
        if source == 'TBE':
            bottom_row_data.append(getattr(tbe_row, col))
        elif source == 'MLP':
            bottom_row_data.append(getattr(mlp_row, col))
        elif source == 'TBE-Recall':
            tbe_combined = getattr(tbe_row, 'All_Coverage_Combined')
            tbe_recall = df_tbe_metrics.loc[i, 'test_sensitivity']
            bottom_row_data.append(tbe_combined * tbe_recall)
        elif source == 'MLP-Recall':
            mlp_combined = getattr(mlp_row, 'All_Coverage_Combined')
            mlp_recall = df_mlp_metrics.loc[i, 'recall']
            bottom_row_data.append(mlp_combined * mlp_recall)
    bottom_heatmap_data.append(bottom_row_data)

# Transpose data to swap axes
top_heatmap_array = np.array(top_heatmap_data).T
bottom_heatmap_array = np.array(bottom_heatmap_data).T

# Generate vertical axis labels and reverse order
top_row_labels = []
for col in reversed(top_columns):  # Reverse column order
    parts = col.split('_')
    dataset = parts[0]
    metric = parts[2]
    top_row_labels.append(f"{dataset}-{metric}-MLP")
    top_row_labels.append(f"{dataset}-{metric}-TBE")

bottom_row_labels = []
for col, source in reversed(bottom_column_order):  # Reverse column order
    if col == 'All_Coverage_Density':
        if source == 'TBE':
            bottom_row_labels.append("All-Density-TBE")
        elif source == 'MLP':
            bottom_row_labels.append("All-Density-MLP")
    elif col == 'All_Coverage_Combined':
        if source == 'TBE':
            bottom_row_labels.append("All-Combined-TBE")
        elif source == 'MLP':
            bottom_row_labels.append("All-Combined-MLP")
        elif source == 'TBE-Recall':
            bottom_row_labels.append("All-Combined×Recall-TBE")
        elif source == 'MLP-Recall':
            bottom_row_labels.append("All-Combined×Recall-MLP")

# Transpose data to swap axes and reverse row order
top_heatmap_array = np.array(top_heatmap_data).T[::-1]  # Reverse row order
bottom_heatmap_array = np.array(bottom_heatmap_data).T[::-1]  # Reverse row order

# Create three-color gradient colormap
colors = ["#ffd3c8", "#85994d", "#3d4d82"]
cmap = LinearSegmentedColormap.from_list("tri_color", colors, N=256)

# Create figure (top-bottom arrangement)
fig = plt.figure(figsize=(15, 10), dpi=300)
gs = fig.add_gridspec(2, 1, height_ratios=[len(top_row_labels), len(bottom_row_labels)], hspace=0.05)

# Top subplot
ax1 = fig.add_subplot(gs[0])
ax1.set_position([0.3, 0.7, 0.8, 0.4])  # [left, bottom, width, height]

# Bottom subplot
ax2 = fig.add_subplot(gs[1])
ax2.set_position([0.3, 0.1, 0.8, 0.4])  # [left, bottom, width, height]

# Set global value range
vmin = 0
vmax = 100

# Plot top heatmap
for i in range(len(top_row_labels)):
    for j in range(len(domain_order)):
        value = top_heatmap_array[i, j]
        size = 0.3 + 0.7 * (value / 100)
        
        rect = Rectangle((j - size/2, i - size/2), size, size,
                         facecolor=cmap(value/100), edgecolor='none')
        ax1.add_patch(rect)
        if value == 0 or value == 100:
            text_val = f"{int(value)}"
        else:
            text_val = f"{value:.1f}"
        
        if value >= 60 :
            color = '#d5d5d5'
        else:
            color='#333333'

        ax1.text(j, i, text_val, ha='center', va='center', 
                color=color, fontsize=10, fontweight='bold')

ax1.set_yticks(np.arange(len(top_row_labels)))
ax1.set_yticklabels(top_row_labels, fontsize=14)
ax1.set_xticks(np.arange(len(domain_order)))
ax1.xaxis.tick_top()
ax1.set_xticklabels(domain_order, rotation=45, ha='left', rotation_mode='anchor', fontsize=14)

ax1.set_xticks(np.arange(len(domain_order)+1)-0.5, minor=True)
ax1.set_yticks(np.arange(len(top_row_labels)+1)-0.5, minor=True)
ax1.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.5)
ax1.tick_params(which='minor', length=0)
ax1.tick_params(axis='both', which='major', length=0, pad=20)

ax1.set_aspect('equal')
ax1.set_xlim(-0.5, len(domain_order)-0.5)
ax1.set_ylim(-0.5, len(top_row_labels)-0.5)

# Plot bottom heatmap
for i in range(len(bottom_row_labels)):
    for j in range(len(domain_order)):
        value = bottom_heatmap_array[i, j]
        size = 0.3 + 0.7 * (value / 100)
        
        # Calculate square position (center in grid center)
        rect = Rectangle((j - size/2, i - size/2), size, size,
                         facecolor=cmap(value/100), edgecolor='none')
        ax2.add_patch(rect)
        
        # Modify: uniformly set to 1 decimal place, 0 and 100 display integers
        if value == 0 or value == 100:
            text_val = f"{int(value)}"
        else:
            text_val = f"{value:.1f}"
        
        if value >= 60 :
            color = '#d5d5d5'
        else:
            color='#333333'

        ax2.text(j, i, text_val, ha='center', va='center', 
                color=color, fontsize=10, fontweight='bold')

ax2.set_yticks(np.arange(len(bottom_row_labels)))
ax2.set_yticklabels(bottom_row_labels, fontsize=14)
ax2.set_xticks(np.arange(len(domain_order)))
ax2.set_xticklabels([])

ax2.set_xticks(np.arange(len(domain_order)+1)-0.5, minor=True)
ax2.set_yticks(np.arange(len(bottom_row_labels)+1)-0.5, minor=True)
ax2.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.5)
ax2.tick_params(which='minor', length=0)
ax2.tick_params(axis='both', which='major', length=0, pad=20)

ax2.set_aspect('equal')
ax2.set_xlim(-0.5, len(domain_order)-0.5)
ax2.set_ylim(-0.5, len(bottom_row_labels)-0.5)

# Add color bar
cax = fig.add_axes([0.92, 0.1, 0.02, 0.85])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('Coverage%', fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=14)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

# Optimize layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.94)

# Save and display
plt.savefig('domain_coverage_square_plot.png', dpi=300, bbox_inches='tight')
plt.show()