"""
Multi-target toxicity risk probability calculator and Visualization
Functionality:
1. Calculate the toxicity risk probability ToxPi from the multi-domain QSAR predictor results and perform weighted integration
Input: multidomainQSAR_predictions.csv
Output: multidomainToxPi_risk.xlsx
2. Visually compare the predicted ToxPi results of the input chemicals with known chemicals
Input: multidomainToxPi_risk.xlsx
Output: comparison_plots
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import List, Dict, Tuple, Any
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Font configuration
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


class MultiTargetRiskCalculator:
    """Multi-target risk probability calculator"""

    def __init__(self):
        # Target info format: {target_name: (similarity, weight)}
        self.target_info = {
            # 机制A: Ecdysis Disruption
            "[A1] EcR_Act": (0.641696974, 0.057954059),
            "[A2] ChitinE_Inh": (0.489617193, 0.057173992),
            
            # 机制B: Neurotoxicity
            "[B1] AChE_Inh": (0.377881185, 0.062145531),
            "[B2] HCN_Inh": (0.690095302, 0.032338746),
            "[B3] T-Ca_Inh": (0.28569777, 0.032338746),
            "[B4] Na_Inh": (0.6582, 0.032338746),
            "[B5] ERG-K_Inh": (0.7059, 0.060108022),
            "[B6] GABA_Act": (0.376824161, 0.04295641),
            "[B7] GluCl_Act": (0.354672647, 0.035605719),
            
            # 机制C: Mitochondrial Toxicity
            "[C1] MitoCom_Inh": (0.807330928, 0.044480067),
            "[C2] MitoOCR_Inc": (0.736091371, 0.062698238),
            "[C3] MitoOCR_Dec": (0.872353891, 0.036672279),
            "[C4] MMP_Dec": (0.720673333, 0.077438708),
            
            # 机制D: Oxidative Stress
            "[D1] NRF1_Act": (0.6635, 0.032942626),
            "[D2] NRF2_Act": (0.4526, 0.039151991),
            "[D3] FoxO_Act": (0.755262288, 0.032942626),
            "[D4] NRF1_Inh": (0.6635, 0.032942626),
            "[D5] NRF2_Inh": (0.452963049, 0.039151991),
            "[D6] FoxO_Inh": (0.763390203, 0.032942626),
            "[D7] antiOX_Inc": (0.798526108, 0.039151991),
            "[D8] antiOX_Dec": (0.885464366, 0.039151991),
            "[D9] HSPA1A_Inc": (0.873914198, 0.028980604),
            "[D10] p53_Inc": (0.2978, 0.025152787),
            "[D11] γH2AX_Inc": (0.8522, 0.023238879),
        }

        # Check weight sum
        total_weight = sum(info[1] for info in self.target_info.values())
        if abs(total_weight - 1.0) > 0.001:
            print(f"Warning: Weight sum is not 1, actual sum: {total_weight:.6f}")

        # Mechanism mapping
        self.mechanism_mapping = {
            'A': '[A] Ecdysis Disruption',
            'B': '[B] Neurotoxicity',
            'C': '[C] Mitochondrial Toxicity',
            'D': '[D] Oxidative Stress'
        }

        # Visualization color configuration
        self.color_config = {
            'Adverse Outcome': '#637038',
            '[A] Ecdysis Disruption': '#576fa0',
            '[B] Neurotoxicity': '#e3b87f',
            '[C] Mitochondrial Toxicity': '#b57979',
            '[D] Oxidative Stress': '#9f9f9f'
        }

        # Mechanism order
        self.mechanism_order = [
            'Adverse Outcome',
            '[A] Ecdysis Disruption',
            '[B] Neurotoxicity',
            '[C] Mitochondrial Toxicity',
            '[D] Oxidative Stress'
        ]

    def extract_target_names(self, columns: List[str]) -> List[str]:
        """Extract target names from column names, excluding _in_ad suffix"""
        target_names = []

        # Find columns ending with _prediction or _confidence
        pattern = re.compile(r'^(.*)_(prediction|confidence)$')

        for col in columns:
            match = pattern.match(col)
            if match:
                target_name = match.group(1)

                # Exclude targets ending with _in_ad
                if not target_name.endswith('_in_ad'):
                    if target_name not in target_names:
                        target_names.append(target_name)

        return target_names

    def get_matching_columns(self, target_name: str, data_columns: List[str]) -> Tuple[str, str]:
        """Get column names matching target name"""
        pred_col = None
        conf_col = None

        for col in data_columns:
            if col == f"{target_name}_prediction":
                pred_col = col
            elif col == f"{target_name}_confidence":
                conf_col = col

        return pred_col, conf_col

    def calculate_risk_probability(self, prediction: Any, confidence: Any) -> float:
        """Calculate risk probability from prediction and confidence"""
        # Handle NA values
        if pd.isna(prediction) or pd.isna(confidence):
            return np.nan

        # Convert to numeric
        try:
            pred_value = float(prediction)
        except (ValueError, TypeError):
            return np.nan

        try:
            conf_value = float(confidence)
        except (ValueError, TypeError):
            return np.nan

        # Calculate risk probability
        if pred_value == 1:
            return conf_value
        elif pred_value == 0:
            return 1.0 - conf_value
        else:
            return np.nan

    def get_mechanism_from_target(self, target_name: str) -> str:
        """Extract mechanism letter from target name"""
        match = re.search(r'\[([A-D])', target_name)
        if match:
            return match.group(1)
        return ''

    def process_data(self, input_file: str) -> Dict[str, pd.DataFrame]:
        """Process data and generate risk assessment results"""
        print(f"Reading data file: {input_file}")

        # Read CSV file
        data = pd.read_csv(input_file)

        # Rename first column to Compound_CID if needed
        if data.columns[0] != 'Compound_CID':
            print(f"Note: Renaming first column to 'Compound_CID'")
            data.rename(columns={data.columns[0]: 'Compound_CID'}, inplace=True)

        # Extract Compound_CID
        compound_ids = data['Compound_CID']

        # Extract target names
        all_columns = data.columns.tolist()
        target_names = self.extract_target_names(all_columns)

        print(f"Found {len(target_names)} targets")

        # Check for missing targets in weight info
        missing_targets = []
        for target in target_names:
            if target not in self.target_info:
                missing_targets.append(target)

        if missing_targets:
            print(f"Warning: {len(missing_targets)} targets not in weight info")

        # Step 1: Calculate risk probability
        print("\nStep 1: Calculating risk probability...")
        risk_prob_results = pd.DataFrame()
        risk_prob_results['Compound_CID'] = compound_ids

        processed_targets = []

        for target in target_names:
            if target in self.target_info:
                pred_col, conf_col = self.get_matching_columns(target, all_columns)

                if pred_col and conf_col:
                    # Calculate risk probability for each row
                    risk_values = []
                    for idx in range(len(data)):
                        pred = data.loc[idx, pred_col]
                        conf = data.loc[idx, conf_col]
                        risk = self.calculate_risk_probability(pred, conf)
                        risk_values.append(risk)

                    risk_prob_results[target] = risk_values
                    processed_targets.append(target)
                else:
                    print(f"Warning: Target {target} missing required columns")

        print(f"Processed {len(processed_targets)} targets")

        # Step 2: Multiply by similarity scores
        print("\nStep 2: Multiplying by similarity scores...")
        similarity_adjusted_results = pd.DataFrame()
        similarity_adjusted_results['Compound_CID'] = compound_ids

        for target in processed_targets:
            if target in risk_prob_results.columns:
                similarity = self.target_info[target][0]
                adjusted_values = risk_prob_results[target] * similarity
                similarity_adjusted_results[target] = adjusted_values

        # Step 3: Weighted integration
        print("\nStep 3: Weighted integration...")
        final_weighted_results = pd.DataFrame()
        final_weighted_results['Compound_CID'] = compound_ids

        # Calculate overall weighted result (Adverse Outcome)
        print("  Calculating overall weighted result...")
        final_weighted_results['Adverse Outcome'] = 0.0

        for target in processed_targets:
            if target in similarity_adjusted_results.columns:
                weight = self.target_info[target][1]
                weighted_values = similarity_adjusted_results[target] * weight
                weighted_values = weighted_values.fillna(0)
                final_weighted_results['Adverse Outcome'] = final_weighted_results['Adverse Outcome'] + weighted_values

        # Calculate weighted results for each mechanism
        print("  Calculating mechanism weighted results...")

        # Group by mechanism
        mechanism_targets = {'A': [], 'B': [], 'C': [], 'D': []}

        for target in processed_targets:
            mechanism = self.get_mechanism_from_target(target)
            if mechanism in mechanism_targets:
                mechanism_targets[mechanism].append(target)

        # Calculate weighted results for each mechanism
        for mechanism, targets in mechanism_targets.items():
            if targets:
                # Calculate weight sum within mechanism
                mechanism_weight_sum = 0
                for target in targets:
                    mechanism_weight_sum += self.target_info[target][1]

                if mechanism_weight_sum > 0:
                    mechanism_column = self.mechanism_mapping.get(mechanism, f'Mechanism_{mechanism}')
                    final_weighted_results[mechanism_column] = 0.0

                    for target in targets:
                        if target in similarity_adjusted_results.columns:
                            weight = self.target_info[target][1]
                            similarity = self.target_info[target][0]

                            # Use normalized weight
                            normalized_weight = weight / mechanism_weight_sum

                            # Calculate weighted value
                            weighted_values = similarity_adjusted_results[target] * normalized_weight
                            weighted_values = weighted_values.fillna(0)

                            final_weighted_results[mechanism_column] = (
                                final_weighted_results[mechanism_column] + weighted_values
                            )

        return {
            'ToxPi_MIEs_KEs': similarity_adjusted_results,
            'ToxPi_Mechanism_AO': final_weighted_results
        }

    def save_to_excel(self, results: Dict[str, pd.DataFrame], output_file: str):
        """Save results to Excel file"""
        print(f"\nSaving results to: {output_file}")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in results.items():
                # Excel sheet names max 31 characters
                valid_sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=valid_sheet_name, index=False)

        print("Results saved successfully!")

    def load_known_chemicals(self, known_file: str) -> Dict[str, pd.DataFrame]:
        """Load known chemicals data"""
        print(f"\nLoading known chemicals data: {known_file}")

        df = pd.read_csv(known_file)
        print(f"Known chemicals data columns: {df.shape[1]}")

        known_data = {}

        # Column indices for each mechanism: (chemical_name_col, value_col, mechanism_name)
        mechanism_columns = [
            (0, 1, 'Adverse Outcome'),
            (3, 4, '[A] Ecdysis Disruption'),
            (6, 7, '[B] Neurotoxicity'),
            (9, 10, '[C] Mitochondrial Toxicity'),
            (12, 13, '[D] Oxidative Stress')
        ]

        for chem_col, val_col, mechanism in mechanism_columns:
            if chem_col < df.shape[1] and val_col < df.shape[1]:
                chemical_names = df.iloc[:, chem_col].astype(str).str.strip()
                values = pd.to_numeric(df.iloc[:, val_col], errors='coerce')

                mechanism_df = pd.DataFrame({
                    'Chemical_Name': chemical_names,
                    'ToxPi_Value': values
                })

                mechanism_df = mechanism_df.dropna()
                known_data[mechanism] = mechanism_df
                print(f"  - {mechanism}: {len(mechanism_df)} chemicals")
            else:
                print(f"Warning: Column index out of range for {mechanism}")

        return known_data

    def create_comparison_plots(self, calculated_df: pd.DataFrame, known_data: Dict[str, pd.DataFrame],
                               output_dir: str = "comparison_plots"):
        """Create comparison plots between known chemicals and calculation results"""
        print(f"\nCreating comparison plots, output directory: {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create plots for each chemical
        for idx, row in calculated_df.iterrows():
            compound_cid = row['Compound_CID']
            if isinstance(compound_cid, float) and compound_cid.is_integer():
                compound_cid_str = str(int(compound_cid))
            else:
                compound_cid_str = str(compound_cid)

            # Create figure
            fig, axes = plt.subplots(1, 5, figsize=(24, 8))
            fig.suptitle(f'Toxicity Risk: Known Chemicals (bar) vs. PubChem CID - {compound_cid_str} (red line)',
                        fontsize=16, fontweight='bold')

            # Adjust subplot layout
            plt.subplots_adjust(left=0.09, bottom=0.06, right=0.90, top=0.90, wspace=0)

            # Add shared axis labels
            fig.text(0.5, 0.02, 'Known Chemicals', ha='center', fontsize=14, fontweight='bold')
            fig.text(0.06, 0.5, 'ToxPi', va='center', rotation='vertical', fontsize=14, fontweight='bold')

            # Create subplot for each mechanism
            for ax_idx, mechanism in enumerate(self.mechanism_order):
                ax = axes[ax_idx]

                if mechanism in known_data:
                    mechanism_df = known_data[mechanism]

                    # Create bar chart
                    x_positions = np.arange(len(mechanism_df))
                    bar_height = 0.6

                    # Set bar color
                    if mechanism in self.color_config:
                        color = self.color_config[mechanism]
                    else:
                        color = '#808080'

                    # Add transparency (60%)
                    color_with_alpha = color + '99'

                    bars = ax.bar(x_positions, mechanism_df['ToxPi_Value'],
                                  color=color_with_alpha, edgecolor=color,
                                  linewidth=1.5, width=bar_height)

                    # Add labels
                    for i, (bar, chem_name) in enumerate(zip(bars, mechanism_df['Chemical_Name'])):
                        height = bar.get_height()

                        # Chemical name label above bar
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008,
                                chem_name, ha='center', va='bottom',
                                fontsize=10, color='black', fontweight='bold',
                                rotation=90)

                        # Value label inside bar
                        ax.text(bar.get_x() + bar.get_width() / 2, height - 0.038,
                                f'{height:.3f}', ha='center', va='center',
                                fontsize=10, color='white', fontweight='bold',
                                rotation=90)

                    # Add calculated reference line
                    if mechanism in row:
                        calc_value = row[mechanism]

                        # Draw horizontal reference line
                        ax.axhline(y=calc_value, color='#be1420', linestyle='--', linewidth=2.5,
                                  label=f'ToxPi: {calc_value:.3f}')

                    # Set subplot title
                    ax.set_title(mechanism, fontsize=14, fontweight='bold')

                    # Remove axis titles
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                    # Set y-axis limits
                    ax.set_ylim([0, 1])

                    # Hide x-axis ticks
                    ax.set_xticks([])

                    # Set y-axis tick label size
                    ax.tick_params(axis='y', labelsize=12)
                    for label in ax.get_yticklabels():
                        label.set_fontweight('bold')

                    # Hide y-axis tick labels for non-first subplots
                    if ax_idx > 0:
                        ax.set_yticklabels([])

                    # Add grid
                    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

                    # Add legend
                    legend = ax.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'}, framealpha=0.9)
                    for text in legend.get_texts():
                        text.set_color("#be1420")

                else:
                    ax.text(0.5, 0.5, f'No data for {mechanism}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(mechanism, fontsize=12, fontweight='bold')

            # Remove space between subplots
            plt.subplots_adjust(wspace=0)

            # Save figure
            output_file = os.path.join(output_dir, f"{compound_cid_str}_vs_known_chemicals.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Created: {output_file}")

        print(f"\nCreated {len(calculated_df)} comparison plots")

    def run_visualization(self, calculated_file: str, known_file: str = "ToxPi_known_chemicals.csv"):
        """Run visualization process"""
        print("=" * 60)
        print("Starting visualization process")
        print("=" * 60)

        # Check if files exist
        if not os.path.exists(known_file):
            print(f"Error: Cannot find known chemicals file '{known_file}'")
            return

        if not os.path.exists(calculated_file):
            print(f"Error: Cannot find calculated results file '{calculated_file}'")
            return

        # Load known chemicals data
        known_data = self.load_known_chemicals(known_file)

        # Load calculated results
        print(f"\nLoading calculated results: {calculated_file}")
        calculated_df = pd.read_excel(calculated_file, sheet_name='ToxPi_Mechanism_AO')

        # Ensure Compound_CID is string
        calculated_df['Compound_CID'] = calculated_df['Compound_CID'].astype(str)

        print(f"Calculated results rows: {len(calculated_df)}")

        # Create comparison plots
        self.create_comparison_plots(calculated_df, known_data)

        print("\nVisualization process completed!")


def main():
    """Main function"""
    print("=" * 60)
    print("Multi-target Risk Probability Calculation and Weighted Integration Tool")
    print("=" * 60)

    # Create calculator instance
    calculator = MultiTargetRiskCalculator()

    # Input file
    input_file = "multidomainQSAR_predictions.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Cannot find input file '{input_file}'")
        print("Please ensure the file is in the current directory")
        return

    # Process data
    results = calculator.process_data(input_file)

    # Output file
    output_file = "multidomainToxPi_risk.xlsx"

    # Save results
    calculator.save_to_excel(results, output_file)

    # Run visualization
    known_file = "ToxPi_known_chemicals.csv"
    calculator.run_visualization(output_file, known_file)

    # Display statistics
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)

    for sheet_name, df in results.items():
        print(f"\nWorksheet: {sheet_name}")
        print(f"  Rows: {df.shape[0]}")
        print(f"  Columns: {df.shape[1]}")
        if 'Compound_CID' in df.columns:
            print(f"  First 5 Compound IDs: {df['Compound_CID'].head().tolist()}")

    print(f"\nOutput file: {output_file}")
    print("Contains two worksheets:")
    print("  1. ToxPi_MIEs_KEs: Adjusted risk probability for each target")
    print("  2. ToxPi_Mechanism_AO: Overall weighted and mechanism weighted results")
    print("\nVisualization results:")
    print("  Comparison plots saved in 'comparison_plots' directory")


if __name__ == "__main__":
    main()