# parse_grid_results.py
"""
Parses the final JSON results from a WE-PaDiM grid search run,
extracts key metrics and parameters, ranks the configurations,
and saves a summary CSV. Optionally prints resource summary info.
"""

import os
import sys
import json
import argparse
import pandas as pd
import glob # To find the results file easily

def parse_results(results_dir: str, model_name: str):
    """
    Parses the grid search results for a given model within a results directory.

    Args:
        results_dir (str): The path to the specific stage/model results directory
                           (e.g., './results/WEPaDiM_Phase1_Reduced/Phase1_ReducedParamSearch_efficientnet-b0_YYYYMMDD_HHMMSS/')
        model_name (str): The name of the model (e.g., 'efficientnet-b0') used to find the correct subdirectory and JSON file.
    """
    print(f"\n--- Parsing Results for Model: {model_name} ---")
    print(f"Searching in directory: {results_dir}")

    # --- Find the main results JSON file ---
    # Path where grid_search.py saves results (within model subdir)
    model_subdir = os.path.join(results_dir, model_name)
    json_pattern = os.path.join(model_subdir, f"{model_name}_wavelet_grid_search_results_final.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"ERROR: Could not find results JSON file matching pattern: {json_pattern}")
        return None
    elif len(json_files) > 1:
        print(f"WARNING: Found multiple results JSON files matching pattern, using the first one: {json_files[0]}")

    results_json_path = json_files[0]
    print(f"Found results file: {results_json_path}")

    # --- Load the main results JSON ---
    try:
        with open(results_json_path, 'r') as f:
            grid_results = json.load(f)
        print(f"Successfully loaded {len(grid_results)} parameter combination results.")
    except Exception as e:
        print(f"ERROR: Failed to load or parse JSON file '{results_json_path}': {e}")
        return None

    if not grid_results:
        print("ERROR: Results list in JSON is empty.")
        return None

    # --- Process the results into a flat list for DataFrame ---
    processed_data = []
    for result_item in grid_results:
        params = result_item.get('params', {})
        # Convert subband list to a sortable string representation
        subbands_str = "_".join(sorted(params.get('wavelet_kept_subbands', [])))

        flat_data = {
            'WaveletType': params.get('wavelet_type', 'N/A'),
            'Level': params.get('wavelet_level', -1),
            'Subbands': subbands_str,
            'Sigma': params.get('sigma', -1.0),
            'CovReg': params.get('cov_reg', -1.0),
            'Dim_Dw': result_item.get('final_feature_dim_dw', -1),
            'FLOPs_G': result_item.get('estimated_flops_g', -1.0),
            'AvgImageAUC': result_item.get('avg_img_auc', 0.0),
            'AvgPixelAUC': result_item.get('avg_pixel_auc', 0.0),
            'AvgPROScore': result_item.get('avg_pro_score', 0.0),
            'AvgTimePerCombo': result_item.get('time', 0.0) # Time for the combo across all classes
            # Add more fields if needed, e.g., average per-class time
        }
        processed_data.append(flat_data)

    # --- Create and Sort DataFrame ---
    if not processed_data:
        print("ERROR: No data processed from the results list.")
        return None

    df = pd.DataFrame(processed_data)

    # Sort by performance metrics (e.g., Image AUC descending, then Pixel AUC descending)
    sort_columns = ['AvgImageAUC', 'AvgPixelAUC', 'AvgPROScore']
    # Check if columns exist before sorting
    valid_sort_columns = [col for col in sort_columns if col in df.columns]
    if not valid_sort_columns:
        print("WARNING: Could not find standard AUC columns for sorting.")
    else:
        df = df.sort_values(by=valid_sort_columns, ascending=[False] * len(valid_sort_columns))

    print(f"\n--- Top 5 Configurations (Sorted by {valid_sort_columns}) ---")
    print(df.head(5).to_string()) # Print top 5 to console

    # --- Save Summary CSV ---
    summary_csv_path = os.path.join(results_dir, f"{model_name}_grid_summary_ranked.csv")
    try:
        df.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"\nRanked summary saved to: {summary_csv_path}")
    except Exception as e:
        print(f"ERROR: Failed to save summary CSV: {e}")
    print(f"\n--- Finished Parsing for Model: {model_name} ---")
    return df # Return the dataframe for potential further use


def main():
    parser = argparse.ArgumentParser(description='Parse WE-PaDiM Grid Search Results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the specific stage results directory containing the model subdirectory (e.g., ./results/WEPaDiM_Phase1_Reduced/Phase1_ReducedParamSearch_efficientnet-b0_YYYYMMDD_HHMMSS/)')
    parser.add_argument('--model', type=str, required=True,
                         help='Name of the model whose results are being parsed (e.g., efficientnet-b0). This is used to find the subdirectory and results file.')

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Provided results directory does not exist: {args.results_dir}")
        sys.exit(1)

    # Check if the model-specific subdirectory exists
    model_subdir_path = os.path.join(args.results_dir, args.model)
    if not os.path.isdir(model_subdir_path):
         print(f"ERROR: Model subdirectory '{args.model}' not found within '{args.results_dir}'")
         print("       Please ensure the --results_dir points to the parent directory created by the runner script (e.g., Stage1_..._YYYYMMDD_HHMMSS).")
         sys.exit(1)


    parse_results(args.results_dir, args.model)


if __name__ == "__main__":
    main()

