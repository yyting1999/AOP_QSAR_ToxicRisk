## QSAR Prediction & Toxicity Risk Assessment Tool - User Guide
This directory provides the trained **Multi-domain MLP prediction model**, its deployment environment, and the **ToxPi risk calculator & visualization tool**. You can directly predict multi-target activity for unknown  chemicals, obtain **Applicability Domain (AD) assessment** results, and then perform toxicity risk assessment (ToxPi) with visual comparison to known reference chemicals.

### Multi-target QSAR Prediction (*A1_MultiDomain_QSAR_Predictor.py*)
- **Prepare Feature File**: Your input must be a CSV file​ and contain all feature columns​ required by the model (*model_required_features.csv*). The input file can be prepared as the sample file (*MultiDomianQSARPredictor_test.csv*).

Ensure the *model_files* folder is in the same directory level as the *A1_MultiDomain_QSAR_Predictor.py* script (i.e., both are inside Model_deployment_predictor/).

- **Execute Prediction**: Run the prediction script

*python A1_MultiDomain_QSAR_Predictor.py --feature_file MultiDomianQSARPredictor_test.csv*

If the script reports missing feature columns, ensure your data file includes all of them. Feature column names must match exactly. If certain features are not measured for some samples, leave the cell empty​ or fill it with NaN​ in the CSV file. The system will automatically perform intelligent imputation using a KNN algorithm based on the training set before prediction.

- **Prediction Output**: The output file (multidomainQSAR_predictions.csv) will contain the prediction and AD assessment results.

### Toxicity Risk Probability (ToxPi) Calculation & Visualization (*A2_ToxPi_Predictor_Risk_Visualization.py*)
- **Input Files**:

*multidomainQSAR_predictions.csv*: The output file from the prediction tool.

*ToxPi_known_chemicals.csv*: Known chemicals reference data file. Contains known chemicals risk probabilities for AO and toxicity mechanisms. If this file is not provided, the tool will only perform calculations without visual comparison.

- **Execute Calculation & Visualization**: Running the ToxPi Tool

*python A2_ToxPi_Predictor_Risk_Visualization.py*

- **Output Results**:

*multidomainToxPi_risk.xlsx*: Calculation results Excel file with two worksheets: *ToxPi_MIEs_KEs*(Risk probability for each target MIEs/KEs). *ToxPi_Mechanism_AO* (Risk probability for [AO] Adverse Outcome; [A] Ecdysis Disruption; [B] Neurotoxicity; [C] Mitochondrial Toxicity and [D] Oxidative Stress.

*comparison_plots*: Generates a 5-subplot comparison chart for each chemical, showing its ToxPi profile alongside known reference chemicals.
