import numpy as np
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import sys
import json
import pandas as pd
from sklearn.metrics import mean_squared_error
from math_methods import apply_poly_regression, apply_exp_regression, poly_reg_str, exp_reg_str, generate_rand_cpu_utils


def get_stats(real_data : List[float], pred_data : List[float]) -> Tuple[Tuple[float, float], float]:
    rmse_error = np.sqrt(mean_squared_error(real_data, pred_data))
    percent_error = (rmse_error / np.max(real_data)) * 100
    corr_matrix = np.corrcoef(real_data, pred_data)
    r_sq = corr_matrix[0,1]**2
    return (rmse_error, percent_error), r_sq

def get_regression_data(lin_reg_dict : Dict[str, Any]) -> Tuple[List[float], float, List[float]]:
    coefs : List[float] = lin_reg_dict["coefficients"]
    r_sq : float = lin_reg_dict["r_squared"]
    std_coef : List[float] = lin_reg_dict["std_coefficients"]
    return coefs, r_sq, std_coef

def generate_watts_from_cpu_util(cpu_utils : List[float], regression_dict : Dict[str, Any]) -> List[float]:
    coefs, r_sq, std_coefs = get_regression_data(regression_dict)

    scaled_cpu_utils : np.ndarray = np.array([x/100 for x in cpu_utils])
    generated_watts : List[float] = list(apply_poly_regression(coefs, scaled_cpu_utils, []))
    #generated_watts : List[float] = list(apply_exp_regression(coefs, scaled_cpu_utils))

    #print(f"Polynomial Regression Eq: {poly_reg_str(coefs)}, R^2: {r_sq:.2f}")
    #print(f"Exponential Regression Eq: {exp_reg_str(coefs)}, R^2: {r_sq:.2f}")
    return generated_watts

def read_cpu_energy_csv(cpu_energy_csv : str) -> Tuple[List[float], List[float]]:
    cpu_energy_df : pd.DataFrame = pd.read_csv(cpu_energy_csv)
    cpu_utils : List[float] = [row["cpu_util"]*100 for _, row in cpu_energy_df.iterrows()]
    watts : List[float] = [row["watts"] for _, row in cpu_energy_df.iterrows()]
    return cpu_utils, watts

def main(cpu_energy_json : str, merged_cpu_energy_csvs : List[str]):

    cpu_energy_dict : Dict[str, Any] = json.load(open(cpu_energy_json, "r"))

    regression_dict = cpu_energy_dict["all_data_linear_reg"]
    
    all_predicted_cpu_utils : List[float] = []
    all_real_cpu_utils : List[float] = []
    all_predicted_watts : List[float] = []
    all_real_watts : List[float] = []

    cpu_percents : List[float] = []
    watts_percents : List[float] = []

    for video_pair_name, pair_dict_data in cpu_energy_dict.items():
        if not video_pair_name.startswith("pair_"):
            continue

        pair_idx = int(video_pair_name.split("_")[1])

        bin_data = pair_dict_data["cpu_bins"]
        bin_ordering = pair_dict_data["cpu_bin_ordering"]
        regression_dict = pair_dict_data["linear_regression"] # If using pair specific regression

        real_cpu_utils, real_watts_list = read_cpu_energy_csv(merged_cpu_energy_csvs[pair_idx-1])

        generated_cpu_utils = generate_rand_cpu_util(bin_data, bin_ordering)
        generated_watts : List[float] = generate_watts_from_cpu_util(generated_cpu_utils, regression_dict)

        (pair_rmse_cpu_error, pair_percent_error_cpu), pair_r_sq_cpu = get_stats(real_cpu_utils, generated_cpu_utils)
        (pair_rmse_watts_error, pair_percent_error_watts), pair_r_sq_watts = get_stats(real_watts_list, generated_watts)

        cpu_percents.append(pair_percent_error_cpu)
        watts_percents.append(pair_percent_error_watts)

        print(f"Video {pair_idx} CPU   RMSE: {pair_rmse_cpu_error:.2f}, {pair_percent_error_cpu:.2f}%, R^2: {pair_r_sq_cpu:.2f}")
        print(f"Video {pair_idx} Watts RMSE: {pair_rmse_watts_error:.2f}, {pair_percent_error_watts:.2f}%, R^2: {pair_r_sq_watts:.2f}")

        print("-"*100)

        # plt.plot(list(range(len(real_cpu_utils))), real_cpu_utils, label="Real CPU Utilization")
        # plt.plot(list(range(len(generated_cpu_utils))), generated_cpu_utils, label="Predicted CPU Utilization")
        # plt.legend()
        # plt.show()

        # plt.plot(list(range(len(real_watts_list))), real_watts_list, label="Real Watts")
        # plt.plot(list(range(len(generated_watts))), generated_watts, label="Predicted Watts")
        # plt.legend()
        # plt.show()

        all_predicted_cpu_utils += generated_cpu_utils
        all_real_cpu_utils += real_cpu_utils
        all_predicted_watts += generated_watts
        all_real_watts += real_watts_list


    (rmse_cpu_error, percent_error_cpu), r_sq_cpu = get_stats(all_real_cpu_utils, all_predicted_cpu_utils)
    (rmse_watts_error, percent_error_watts), r_sq_watts = get_stats(all_real_watts, all_predicted_watts)

    print(f"Overall CPU   RMSE: {rmse_cpu_error:.2f}, {percent_error_cpu:.2f}%, R^2: {r_sq_cpu:.2f}")
    print(f"Overall Watts RMSE: {rmse_watts_error:.2f}, {percent_error_watts:.2f}%, R^2: {r_sq_watts:.2f}")
    print(f"Average CPU Percent: {np.mean(cpu_percents):.2f}%, Average Watts Percent: {np.mean(watts_percents):.2f}%")
    
    # plt.plot(list(range(len(all_real_cpu_utils))), all_real_cpu_utils, label="Real CPU Utilization")
    # plt.plot(list(range(len(all_predicted_cpu_utils))), all_predicted_cpu_utils, label="Predicted CPU Utilization")
    # plt.legend()
    # plt.show()

    # plt.plot(list(range(len(all_real_watts))), all_real_watts, label="Real Watts")
    # plt.plot(list(range(len(all_predicted_watts))), all_predicted_watts, label="Predicted Watts")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 test_cpu_energy_accuracy.py <cpu_energy.json> <merged_cpu_energy.csv> * n")
        exit(1)
    args = sys.argv[1:]
    cpu_energy_json_path = args[0]
    merged_cpu_energy_csv_paths = args[1:]
    if cpu_energy_json_path[-5:] != ".json":
        print("Usage: python3 test_cpu_energy_accuracy.py <cpu_energy.json> <merged_cpu_energy.csv> * n")
        exit(1)

    for path in merged_cpu_energy_csv_paths:
        if path[-4:] != ".csv":
            print("Usage: python3 test_cpu_energy_accuracy.py <cpu_energy.json> <merged_cpu_energy.csv> * n")
            exit(1)

    main(cpu_energy_json_path, merged_cpu_energy_csv_paths)


# -------------------------